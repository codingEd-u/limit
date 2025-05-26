import os
from typing import Any

# Monkeypatch coverage to bypass teardown crash in act/docker
if os.getenv("COVERAGE_PROCESS_START"):
    import coverage

    coverage.process_startup()

    # Nukes the teardown assertion that fails in act
    import coverage.collector

    def safe_stop(self: Any) -> None:
        if self in getattr(self, "_collectors", []):
            self._collectors.remove(self)

    coverage.collector.Collector.stop = safe_stop
