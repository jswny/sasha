from __future__ import annotations

import json
import sqlite3
from typing import Any, Iterable, Optional, Tuple


class ArenaDB:
    def __init__(self, path: str) -> None:
        self.path = path
        self.conn = sqlite3.connect(self.path)
        self.conn.row_factory = sqlite3.Row
        self._configure()
        self._init_schema()

    def _configure(self) -> None:
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute("PRAGMA foreign_keys=ON")
        self.conn.execute("PRAGMA busy_timeout=5000")

    def _init_schema(self) -> None:
        current_version = self.conn.execute("PRAGMA user_version").fetchone()[0]
        if current_version < 2:
            self.conn.execute("DROP TABLE IF EXISTS decisions")
            self.conn.execute("DROP TABLE IF EXISTS actions")
            self.conn.execute("DROP TABLE IF EXISTS hands")
            self.conn.execute("DROP TABLE IF EXISTS runs")
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS runs (
                run_id TEXT PRIMARY KEY,
                created_at TEXT NOT NULL,
                matchup_name TEXT NOT NULL,
                model_a_id TEXT NOT NULL,
                model_b_id TEXT NOT NULL,
                memory_mode_a INTEGER NOT NULL,
                memory_mode_b INTEGER NOT NULL,
                rng_seed INTEGER NOT NULL,
                blinds TEXT NOT NULL,
                starting_stack_bb INTEGER NOT NULL,
                code_version TEXT,
                next_hand_index INTEGER NOT NULL DEFAULT 0,
                hand_count INTEGER NOT NULL,
                config_json TEXT
            )
            """
        )
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS hands (
                hand_id TEXT PRIMARY KEY,
                run_id TEXT NOT NULL,
                hand_index INTEGER NOT NULL,
                button_seat INTEGER NOT NULL,
                starting_stacks_json TEXT NOT NULL,
                ending_stacks_json TEXT NOT NULL,
                board_cards_json TEXT NOT NULL,
                winner_seat INTEGER,
                pot_final INTEGER NOT NULL,
                showdown_json TEXT,
                UNIQUE(run_id, hand_index),
                FOREIGN KEY(run_id) REFERENCES runs(run_id)
            )
            """
        )
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS actions (
                action_id TEXT PRIMARY KEY,
                hand_id TEXT NOT NULL,
                street TEXT NOT NULL,
                action_index INTEGER NOT NULL,
                actor_seat INTEGER NOT NULL,
                action_type TEXT NOT NULL,
                action_amount INTEGER NOT NULL,
                to_call INTEGER NOT NULL,
                min_raise INTEGER,
                max_raise INTEGER,
                legal_actions_json TEXT NOT NULL,
                pot_before INTEGER NOT NULL,
                pot_after INTEGER NOT NULL,
                stacks_before_json TEXT NOT NULL,
                stacks_after_json TEXT NOT NULL,
                board_snapshot_json TEXT NOT NULL,
                UNIQUE(hand_id, action_index),
                FOREIGN KEY(hand_id) REFERENCES hands(hand_id)
            )
            """
        )
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS decisions (
                action_id TEXT PRIMARY KEY,
                model_id TEXT NOT NULL,
                prompt_text TEXT NOT NULL,
                response_json TEXT NOT NULL,
                memory_summary TEXT,
                rationale TEXT,
                latency_ms INTEGER,
                token_usage_json TEXT,
                retry_count INTEGER NOT NULL,
                error_type TEXT,
                FOREIGN KEY(action_id) REFERENCES actions(action_id)
            )
            """
        )
        self.conn.execute("PRAGMA user_version=2")
        self.conn.commit()

    def close(self) -> None:
        self.conn.close()

    def create_run(self, *, run_id: str, created_at: str, matchup_name: str, model_a_id: str, model_b_id: str,
                   memory_mode_a: bool, memory_mode_b: bool, rng_seed: int, blinds: Tuple[int, int],
                   starting_stack_bb: int, code_version: str, hand_count: int,
                   config_json: Optional[str]) -> None:
        self.conn.execute(
            """
            INSERT INTO runs (
                run_id, created_at, matchup_name, model_a_id, model_b_id, memory_mode_a, memory_mode_b,
                rng_seed, blinds, starting_stack_bb, code_version,
                next_hand_index, hand_count, config_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 0, ?, ?)
            """,
            (
                run_id,
                created_at,
                matchup_name,
                model_a_id,
                model_b_id,
                1 if memory_mode_a else 0,
                1 if memory_mode_b else 0,
                rng_seed,
                json.dumps(list(blinds)),
                starting_stack_bb,
                code_version,
                hand_count,
                config_json,
            ),
        )
        self.conn.commit()

    def get_run(self, run_id: str) -> Optional[sqlite3.Row]:
        cur = self.conn.execute("SELECT * FROM runs WHERE run_id = ?", (run_id,))
        return cur.fetchone()

    def lease_hand_range(self, run_id: str, batch_size: int) -> Optional[Tuple[int, int]]:
        self.conn.execute("BEGIN IMMEDIATE")
        row = self.conn.execute(
            "SELECT next_hand_index, hand_count FROM runs WHERE run_id = ?",
            (run_id,),
        ).fetchone()
        if row is None:
            self.conn.execute("ROLLBACK")
            raise ValueError(f"Run not found: {run_id}")
        start = int(row["next_hand_index"])
        hand_count = int(row["hand_count"])
        if start >= hand_count:
            self.conn.execute("ROLLBACK")
            return None
        end = min(hand_count, start + batch_size)
        self.conn.execute(
            "UPDATE runs SET next_hand_index = ? WHERE run_id = ?",
            (end, run_id),
        )
        self.conn.execute("COMMIT")
        return start, end

    def insert_hand(self, **kwargs: Any) -> None:
        self.conn.execute(
            """
            INSERT INTO hands (
                hand_id, run_id, hand_index, button_seat, starting_stacks_json, ending_stacks_json,
                board_cards_json, winner_seat, pot_final, showdown_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                kwargs["hand_id"],
                kwargs["run_id"],
                kwargs["hand_index"],
                kwargs["button_seat"],
                kwargs["starting_stacks_json"],
                kwargs["ending_stacks_json"],
                kwargs["board_cards_json"],
                kwargs.get("winner_seat"),
                kwargs["pot_final"],
                kwargs.get("showdown_json"),
            ),
        )
        self.conn.commit()

    def insert_hand_start(self, *, hand_id: str, run_id: str, hand_index: int, button_seat: int, starting_stacks_json: str) -> None:
        self.conn.execute(
            """
            INSERT INTO hands (
                hand_id, run_id, hand_index, button_seat, starting_stacks_json, ending_stacks_json,
                board_cards_json, winner_seat, pot_final, showdown_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                hand_id,
                run_id,
                hand_index,
                button_seat,
                starting_stacks_json,
                starting_stacks_json,
                json.dumps([]),
                None,
                0,
                None,
            ),
        )
        self.conn.commit()

    def update_hand_final(
        self,
        *,
        hand_id: str,
        ending_stacks_json: str,
        board_cards_json: str,
        winner_seat: Optional[int],
        pot_final: int,
        showdown_json: Optional[str],
    ) -> None:
        self.conn.execute(
            """
            UPDATE hands
            SET ending_stacks_json = ?, board_cards_json = ?, winner_seat = ?, pot_final = ?, showdown_json = ?
            WHERE hand_id = ?
            """,
            (
                ending_stacks_json,
                board_cards_json,
                winner_seat,
                pot_final,
                showdown_json,
                hand_id,
            ),
        )
        self.conn.commit()

    def insert_action(self, **kwargs: Any) -> None:
        self.conn.execute(
            """
            INSERT INTO actions (
                action_id, hand_id, street, action_index, actor_seat, action_type, action_amount,
                to_call, min_raise, max_raise, legal_actions_json, pot_before, pot_after,
                stacks_before_json, stacks_after_json, board_snapshot_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                kwargs["action_id"],
                kwargs["hand_id"],
                kwargs["street"],
                kwargs["action_index"],
                kwargs["actor_seat"],
                kwargs["action_type"],
                kwargs["action_amount"],
                kwargs["to_call"],
                kwargs.get("min_raise"),
                kwargs.get("max_raise"),
                kwargs["legal_actions_json"],
                kwargs["pot_before"],
                kwargs["pot_after"],
                kwargs["stacks_before_json"],
                kwargs["stacks_after_json"],
                kwargs["board_snapshot_json"],
            ),
        )
        self.conn.commit()

    def insert_decision(self, **kwargs: Any) -> None:
        self.conn.execute(
            """
            INSERT INTO decisions (
                action_id, model_id, prompt_text, response_json, memory_summary, rationale,
                latency_ms, token_usage_json, retry_count, error_type
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                kwargs["action_id"],
                kwargs["model_id"],
                kwargs["prompt_text"],
                kwargs["response_json"],
                kwargs.get("memory_summary"),
                kwargs.get("rationale"),
                kwargs.get("latency_ms"),
                kwargs.get("token_usage_json"),
                kwargs["retry_count"],
                kwargs.get("error_type"),
            ),
        )
        self.conn.commit()
