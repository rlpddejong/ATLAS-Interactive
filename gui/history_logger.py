import json
import os
from datetime import datetime, timezone


class HistoryLogger:
    """Appends timestamped annotation events to a JSON file in the workspace.

    Stored as a single JSON array (rather than JSON Lines) so it stays a plain
    file people can open and inspect directly. Written atomically so a crash
    mid-write can't corrupt previously logged history.
    """

    def __init__(self, workspace: str):
        self.path = os.path.join(workspace, 'history.json')
        self.events = self._load_existing()

    def _load_existing(self):
        if os.path.exists(self.path):
            try:
                with open(self.path, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except (json.JSONDecodeError, OSError):
                return []
        return []

    def log(self, event: str, **fields):
        record = {
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'event': event,
            **fields,
        }
        self.events.append(record)
        self._write()

    def _write(self):
        tmp_path = self.path + '.tmp'
        with open(tmp_path, 'w', encoding='utf-8') as f:
            json.dump(self.events, f, indent=2)
        os.replace(tmp_path, self.path)
