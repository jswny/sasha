from __future__ import annotations

import json
import logging
import random
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple

from pokerkit import Automation, Mode, NoLimitTexasHoldem, State

from .cards import cards_to_shorthand
from .config import AgentConfig, ArenaConfig, MatchupConfig, RunConfig
from .db import ArenaDB
from .llm import LLMClient, LLMCallResult, LLMAgent
from .prompt import build_messages
from .rng import stable_seed


@dataclass
class HistoryEntry:
    street: str
    text: str
    action_type: str
    amount: int


def _street_label(state: State) -> str:
    idx = state.street_index
    if idx == 0:
        return "Preflop"
    if idx == 1:
        return "Flop"
    if idx == 2:
        return "Turn"
    if idx == 3:
        return "River"
    return "Unknown"


def _legal_actions(state: State) -> List[str]:
    actions: List[str] = []
    to_call = state.checking_or_calling_amount
    if state.can_fold() and to_call > 0:
        actions.append("fold")
    if state.can_check_or_call():
        actions.append("check" if to_call == 0 else "call")
    if state.can_complete_bet_or_raise_to():
        actions.append("bet" if to_call == 0 else "raise")
    return actions


def _validate_decision(decision, state: State) -> Tuple[bool, Optional[str]]:
    to_call = state.checking_or_calling_amount
    legal = _legal_actions(state)
    action_type = decision.action_type

    if action_type == "fold" and to_call == 0:
        return False, "fold not allowed when to_call == 0"

    if action_type not in legal:
        return False, f"action_type {action_type} not in legal actions {legal}"

    if action_type in {"fold", "check", "call"}:
        if decision.action_amount != 0:
            return False, "action_amount must be 0 for fold/check/call"
        if action_type == "check" and to_call != 0:
            return False, "check is not legal when to_call > 0"
        if action_type == "call" and to_call == 0:
            return False, "call is not legal when to_call == 0"
        return True, None

    if action_type in {"bet", "raise"}:
        if not state.can_complete_bet_or_raise_to():
            return False, "bet/raise not allowed"
        min_raise = state.min_completion_betting_or_raising_to_amount
        max_raise = state.max_completion_betting_or_raising_to_amount
        if min_raise is None or max_raise is None:
            return False, "min/max raise not available"
        if not (min_raise <= decision.action_amount <= max_raise):
            return False, f"action_amount {decision.action_amount} not in [{min_raise}, {max_raise}]"
        return True, None

    return False, "unknown action_type"


def _apply_action(state: State, action_type: str, action_amount: int) -> None:
    if action_type == "fold":
        state.fold()
        return
    if action_type in {"check", "call"}:
        state.check_or_call()
        return
    if action_type in {"bet", "raise"}:
        state.complete_bet_or_raise_to(action_amount)
        return
    raise ValueError(f"Unsupported action_type: {action_type}")


def _fallback_action(state: State) -> Tuple[str, int]:
    if state.can_check_or_call():
        return ("check" if state.checking_or_calling_amount == 0 else "call"), 0
    if state.can_fold():
        return "fold", 0
    if state.can_complete_bet_or_raise_to():
        min_raise = state.min_completion_betting_or_raising_to_amount
        return "raise", min_raise if min_raise is not None else 0
    return "fold", 0


def _action_text(action_type: str, amount: int) -> str:
    if action_type == "fold":
        return "folds"
    if action_type == "check":
        return "checks"
    if action_type == "call":
        return f"calls {amount}"
    if action_type == "bet":
        return f"bets {amount}"
    if action_type == "raise":
        return f"raises to {amount}"
    return action_type


def _build_state(
    *,
    blinds: Tuple[int, int],
    starting_stack: int,
    rng_seed: int,
) -> State:
    random.seed(rng_seed)
    return NoLimitTexasHoldem.create_state(
        (
            Automation.ANTE_POSTING,
            Automation.BET_COLLECTION,
            Automation.BLIND_OR_STRADDLE_POSTING,
            Automation.CARD_BURNING,
            Automation.HOLE_DEALING,
            Automation.BOARD_DEALING,
            Automation.HOLE_CARDS_SHOWING_OR_MUCKING,
            Automation.HAND_KILLING,
            Automation.CHIPS_PUSHING,
            Automation.CHIPS_PULLING,
        ),
        False,
        0,
        blinds,
        blinds[1],
        (starting_stack, starting_stack),
        2,
        mode=Mode.CASH_GAME,
    )


def _seat_mapping(hand_index: int, agent_a: LLMAgent, agent_b: LLMAgent) -> Dict[int, LLMAgent]:
    if hand_index % 2 == 0:
        return {0: agent_a, 1: agent_b}
    return {0: agent_b, 1: agent_a}


def _find_agent_cfg(agent_id: str, agents: List[AgentConfig]) -> AgentConfig:
    for agent in agents:
        if agent.id == agent_id:
            return agent
    raise ValueError(f"Agent not found: {agent_id}")


def _prepare_matchups(config: ArenaConfig) -> List[MatchupConfig]:
    if config.matchups:
        return config.matchups
    matchups: List[MatchupConfig] = []
    for i, agent_a in enumerate(config.agents):
        for agent_b in config.agents[i + 1 :]:
            name = f"{agent_a.id}_vs_{agent_b.id}"
            matchups.append(MatchupConfig(name=name, a=agent_a.id, b=agent_b.id))
    return matchups


def run_arena(config: ArenaConfig) -> None:
    db = ArenaDB(config.run.db_path)
    llm_client = LLMClient(config.openrouter)

    try:
        for matchup in _prepare_matchups(config):
            agent_a_cfg = _find_agent_cfg(matchup.a, config.agents)
            agent_b_cfg = _find_agent_cfg(matchup.b, config.agents)
            agent_a = llm_client.build_agent(agent_a_cfg)
            agent_b = llm_client.build_agent(agent_b_cfg)
            run_id = _init_run(db, config.run, matchup, agent_a_cfg, agent_b_cfg)
            _run_matchup(
                db,
                config.run,
                matchup,
                run_id,
                agent_a,
                agent_b,
            )
    finally:
        db.close()


def _init_run(
    db: ArenaDB,
    run_cfg: RunConfig,
    matchup: MatchupConfig,
    agent_a_cfg: AgentConfig,
    agent_b_cfg: AgentConfig,
) -> str:
    if run_cfg.run_id:
        existing = db.get_run(run_cfg.run_id)
        if existing is None:
            raise ValueError(f"Run id {run_cfg.run_id} not found")
        return run_cfg.run_id

    run_id = uuid.uuid4().hex
    created_at = datetime.now(timezone.utc).isoformat()
    db.create_run(
        run_id=run_id,
        created_at=created_at,
        matchup_name=matchup.name,
        model_a_id=agent_a_cfg.model,
        model_b_id=agent_b_cfg.model,
        memory_mode_a=agent_a_cfg.use_memory,
        memory_mode_b=agent_b_cfg.use_memory,
        rng_seed=run_cfg.run_seed,
        blinds=run_cfg.blinds,
        starting_stack_bb=run_cfg.starting_stack_bb,
        code_version=run_cfg.code_version,
        hand_count=run_cfg.hand_count,
        config_json=None,
    )
    return run_id


def _run_matchup(
    db: ArenaDB,
    run_cfg: RunConfig,
    matchup: MatchupConfig,
    run_id: str,
    agent_a: LLMAgent,
    agent_b: LLMAgent,
) -> None:
    batch_size = max(1, run_cfg.lease_batch_size)
    while True:
        hand_range = db.lease_hand_range(run_id, batch_size)
        if hand_range is None:
            break
        start, end = hand_range
        for hand_index in range(start, end):
            _play_hand(
                db=db,
                run_cfg=run_cfg,
                matchup=matchup,
                run_id=run_id,
                hand_index=hand_index,
                agent_a=agent_a,
                agent_b=agent_b,
            )


def _play_hand(
    *,
    db: ArenaDB,
    run_cfg: RunConfig,
    matchup: MatchupConfig,
    run_id: str,
    hand_index: int,
    agent_a: LLMAgent,
    agent_b: LLMAgent,
) -> None:
    logger = logging.getLogger("sasha.arena")
    agent_a.reset_hand_memory()
    agent_b.reset_hand_memory()

    rng_seed = stable_seed(run_cfg.run_seed, hand_index, "deck")
    starting_stack = run_cfg.starting_stack_bb * run_cfg.blinds[1]
    state = _build_state(blinds=run_cfg.blinds, starting_stack=starting_stack, rng_seed=rng_seed)

    seat_to_agent = _seat_mapping(hand_index, agent_a, agent_b)
    button_seat = 1  # heads-up: seat 1 is the small blind/button in PokerKit
    hand_id = f"{run_id}:{hand_index}"
    starting_stacks = [starting_stack, starting_stack]
    db.insert_hand_start(
        hand_id=hand_id,
        run_id=run_id,
        hand_index=hand_index,
        button_seat=button_seat,
        starting_stacks_json=json.dumps(starting_stacks),
    )

    history: List[HistoryEntry] = []
    action_index = 0
    max_pot = state.total_pot_amount

    while state.status:
        if state.actor_index is None:
            if state.can_no_operate():
                state.no_operate()
                continue
            break

        actor_seat = state.actor_index
        acting_agent = seat_to_agent[actor_seat]

        street = _street_label(state)
        board_cards = list(state.get_board_cards(0))
        hole_cards = list(state.hole_cards[actor_seat])
        to_call = state.checking_or_calling_amount
        min_raise = state.min_completion_betting_or_raising_to_amount
        max_raise = state.max_completion_betting_or_raising_to_amount
        legal_actions = _legal_actions(state)

        prompt_messages = build_messages(
            hand_id=hand_id,
            street=street,
            button_seat=button_seat,
            blinds=run_cfg.blinds,
            starting_stacks=starting_stacks,
            current_stacks=list(state.stacks),
            pot=state.total_pot_amount,
            board_cards=board_cards,
            hole_cards=hole_cards,
            history_actions=[
                {
                    "street": entry.street,
                    "text": entry.text,
                }
                for entry in history
            ],
            to_call=to_call,
            min_raise=min_raise,
            max_raise=max_raise,
            legal_actions=legal_actions,
            memory_summary=acting_agent.get_memory_for_prompt(),
        )

        decision_result, error_type = _get_valid_decision(
            agent=acting_agent,
            prompt_messages=prompt_messages,
            state=state,
            hand_id=hand_id,
        )

        if decision_result.decision is None:
            action_type, action_amount = _fallback_action(state)
            error_type = error_type or "Fallback"
            logger.error(
                "Fallback action used (hand_id=%s, street=%s, actor=%s, error=%s)",
                hand_id,
                street,
                actor_seat,
                error_type,
            )
        else:
            action_type = decision_result.decision.action_type
            action_amount = decision_result.decision.action_amount

        pot_before = state.total_pot_amount
        stacks_before = list(state.stacks)
        board_snapshot = cards_to_shorthand(board_cards)

        _apply_action(state, action_type, action_amount)

        pot_after = state.total_pot_amount
        stacks_after = list(state.stacks)
        if pot_after > max_pot:
            max_pot = pot_after

        display_amount = action_amount
        if action_type == "call":
            display_amount = to_call
        action_text = _action_text(action_type, display_amount)
        history.append(
            HistoryEntry(
                street=street,
                text=f"Player {actor_seat + 1} {action_text}",
                action_type=action_type,
                amount=action_amount,
            )
        )

        action_id = f"{run_id}:{hand_index}:{action_index}"

        db.insert_action(
            action_id=action_id,
            hand_id=hand_id,
            street=street,
            action_index=action_index,
            actor_seat=actor_seat,
            action_type=action_type,
            action_amount=action_amount,
            to_call=to_call,
            min_raise=min_raise,
            max_raise=max_raise,
            legal_actions_json=json.dumps(legal_actions),
            pot_before=pot_before,
            pot_after=pot_after,
            stacks_before_json=json.dumps(stacks_before),
            stacks_after_json=json.dumps(stacks_after),
            board_snapshot_json=json.dumps(board_snapshot),
        )

        db.insert_decision(
            action_id=action_id,
            model_id=acting_agent.agent_cfg.model,
            prompt_text=json.dumps(prompt_messages),
            response_json=decision_result.response_json or "{}",
            memory_summary=json.dumps(
                decision_result.decision.memory_summary
                if decision_result.decision is not None
                else []
            ),
            rationale=(
                decision_result.decision.rationale
                if decision_result.decision is not None
                else None
            ),
            latency_ms=decision_result.latency_ms,
            token_usage_json=decision_result.token_usage_json,
            retry_count=decision_result.retry_count,
            error_type=error_type,
        )

        if decision_result.decision is not None:
            acting_agent.update_memory(decision_result.decision)

        action_index += 1

    ending_stacks = list(state.stacks)
    board_cards_final = cards_to_shorthand(list(state.get_board_cards(0)))

    max_stack = max(ending_stacks)
    winners = [i for i, s in enumerate(ending_stacks) if s == max_stack]
    winner_seat = winners[0] if len(winners) == 1 else None

    showdown = {
        "hole_cards": [cards_to_shorthand(cards) for cards in state.hole_cards],
    }

    db.update_hand_final(
        hand_id=hand_id,
        ending_stacks_json=json.dumps(ending_stacks),
        board_cards_json=json.dumps(board_cards_final),
        winner_seat=winner_seat,
        pot_final=max_pot,
        showdown_json=json.dumps(showdown),
    )


def _get_valid_decision(
    *,
    agent: LLMAgent,
    prompt_messages: List[dict],
    state: State,
    hand_id: str,
) -> Tuple[LLMCallResult, Optional[str]]:
    logger = logging.getLogger("sasha.arena")
    result = agent.decide(messages=prompt_messages)

    if result.decision is None:
        error_type = result.error_type or "ProviderError"
        return result, error_type

    valid, reason = _validate_decision(result.decision, state)
    if valid:
        return result, None

    logger.warning(
        "Invalid action from model (hand_id=%s, street=%s, actor=%s, reason=%s)",
        hand_id,
        _street_label(state),
        state.actor_index,
        reason,
    )
    return result, "InvalidAction"
