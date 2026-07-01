from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from miloco.automation.schema import MiotEventMapping, MiotEventTrigger
from miloco.automation.service import (
    AutomationService,
    _coerce_number,
    _match_condition,
)
from miloco.perception.types import CaptionEntry, MatchedRule
from miloco.rule.schema import RuleTriggerType


class _KVRepoStub:
    def __init__(self) -> None:
        self._store: dict[str, str] = {}

    def get(self, key: str, default: str | None = None) -> str | None:
        return self._store.get(key, default)

    def set(self, key: str, value: str) -> None:
        self._store[key] = value


@pytest.mark.parametrize(
    ("actual", "expected", "matched"),
    [
        ("1", {"op": "eq", "value": "1"}, True),
        ("1", {"op": "ne", "value": "0"}, True),
        ("12", {"op": "gt", "value": "10"}, True),
        ("3", {"op": "lt", "value": "5"}, True),
        ("9", {"op": "gte", "value": "9"}, True),
        ("9", {"op": "lte", "value": "9"}, True),
        ("abc", {"op": "gt", "value": "1"}, False),
        (None, {"op": "any", "value": "*"}, False),
        (None, {"op": "eq", "value": "None"}, False),
    ],
)
def test_match_condition_supports_string_and_numeric_operators(
    actual,
    expected,
    matched,
):
    assert _match_condition(actual, expected) is matched


@pytest.mark.parametrize(
    ("value", "number"),
    [
        ("12", 12.0),
        (" 7.5 ", 7.5),
        (8, 8.0),
        (True, None),
        ("abc", None),
    ],
)
def test_coerce_number(value, number):
    assert _coerce_number(value) == number


@pytest.mark.asyncio
async def test_handle_trigger_uses_structured_perception_context():
    service = AutomationService(_KVRepoStub())
    service.create_mapping(
        MiotEventMapping(
            source_type="device",
            source_id="sensor-1",
            source_name_snapshot="门磁",
            camera_dids=["cam-1"],
            enabled=True,
            query_template="重点看门口",
            event_kinds=["device_prop"],
            property_filters={"prop.2.1": {"op": "eq", "value": "1"}},
            cooldown_seconds=0,
        )
    )

    captured: dict[str, object] = {}

    async def _structured(sources, rules, extra_context="", snapshot_sink=None):
        captured["sources"] = sources
        captured["rules"] = rules
        captured["extra_context"] = extra_context
        captured["snapshot_sink"] = snapshot_sink
        return SimpleNamespace(
            caption=[SimpleNamespace(description="门口无人")],
            suggestions=[],
            matched_rules=[],
        )

    perception_service = SimpleNamespace(
        structured_on_demand_perceive=_structured,
        publish_meaningful_event=lambda _: None,
    )
    rule_service = SimpleNamespace(get_all_rules=AsyncMock(return_value=[]))
    meaningful_events_dao = SimpleNamespace(
        insert=lambda **_: None,
        update_snapshot_count=lambda *_: None,
    )

    trigger = MiotEventTrigger(
        source_type="device",
        source_id="sensor-1",
        source_name="门磁",
        event_name="device_prop",
        changed_properties={"prop.2.1": "1"},
        occurred_at=1234567890,
        raw={},
    )

    await service.handle_trigger(
        trigger=trigger,
        perception_service=perception_service,
        rule_service=rule_service,
        miot_service=None,
        meaningful_events_dao=meaningful_events_dao,
    )

    assert captured["sources"] == ["cam-1"]
    assert "米家触发上下文" in captured["extra_context"]
    assert "属性变化" in captured["extra_context"]
    assert captured["rules"][0]["condition"]["query"] == "重点看门口"


@pytest.mark.asyncio
async def test_handle_trigger_updates_only_structured_real_rule_matches():
    service = AutomationService(_KVRepoStub())
    mapping = service.create_mapping(
        MiotEventMapping(
            source_type="device",
            source_id="sensor-1",
            source_name_snapshot="门磁",
            camera_dids=["cam-1"],
            enabled=True,
            query_template="重点看门口",
            event_kinds=["device_prop"],
            property_filters={"prop.2.1": {"op": "eq", "value": "1"}},
            cooldown_seconds=0,
        )
    )

    rule = SimpleNamespace(
        id="rule-1",
        name="门口有人",
        trigger_type=RuleTriggerType.MIOT_EVENT,
        condition=SimpleNamespace(
            source_ids=["sensor-1"],
            event_kinds=["device_prop"],
            property_filters={"prop.2.1": {"op": "eq", "value": "1"}},
            mapping_ids=[],
            use_global_mapping=True,
            query="画面中是否有人在门口",
        ),
    )
    update_state = AsyncMock()

    async def _structured(*_, **__):
        return SimpleNamespace(
            caption=[CaptionEntry(description="门口有人")],
            suggestions=[],
            matched_rules=[
                MatchedRule(
                    rule_id="rule-1",
                    rule_name="门口有人",
                    reason="画面中有人在门口",
                    source_device_ids=["cam-1"],
                ),
                MatchedRule(
                    rule_id=f"miot_mapping:{mapping.id}",
                    rule_name="[感知触发] 门磁",
                    reason="临时感知提示命中",
                    source_device_ids=["cam-1"],
                ),
            ],
        )

    perception_service = SimpleNamespace(
        structured_on_demand_perceive=_structured,
        publish_meaningful_event=lambda _: None,
    )
    rule_service = SimpleNamespace(
        get_all_rules=AsyncMock(return_value=[rule]),
        update_state=update_state,
    )
    meaningful_events_dao = SimpleNamespace(
        insert=lambda **_: None,
        update_snapshot_count=lambda *_: None,
    )

    log = await service.handle_trigger(
        trigger=MiotEventTrigger(
            source_type="device",
            source_id="sensor-1",
            source_name="门磁",
            event_name="device_prop",
            changed_properties={"prop.2.1": "1"},
            occurred_at=1234567890,
            raw={},
        ),
        perception_service=perception_service,
        rule_service=rule_service,
        miot_service=None,
        meaningful_events_dao=meaningful_events_dao,
    )

    update_state.assert_awaited_once()
    assert update_state.await_args.args[:4] == (
        "rule-1",
        "cam-1",
        True,
        "画面中有人在门口",
    )
    assert log.matched_rule_ids == ["rule-1", f"miot_mapping:{mapping.id}"]
    assert len(log.structured_matched_rules) == 2
