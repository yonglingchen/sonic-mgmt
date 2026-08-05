# flake8: noqa: F403, F401, F405
import pytest
import logging

from snappi_tests.rocev2.files.helper import *
from snappi_tests.rocev2.files.helper_tc2_portable import collect_flow_queue_stats_portable
from snappi_tests.rocev2.files.helper_topo2 import (
    build_clos_ecmp_topology,
    ecmp_hash_verify,
)

logger = logging.getLogger(__name__)

pytestmark = [pytest.mark.topology('multidut-tgen')]

TRAFFIC_DURATION = 120
CNP_DSCP = 48
TX_RATE_PERCENT = 30   # 30% per rank keeps aggregate below 100% across Spine


def test_rocev2_tc7_ecmp_hashing(
                                    request,
                                    snappi_api,
                                    conn_graph_facts,
                                    fanout_graph_facts_multidut,
                                    get_snappi_ports,
                                    duthosts,
                                    prio_dscp_map,
                                ):
    """
    TC7 - ECMP Hashing / Load Balancing across Spine.

    Topology 2: 2 Leaf + 1 Spine Clos, 8 tgen ports.
      Leaf0 server-facing: tgen_port0..tgen_port3 (source ranks)
      Leaf1 server-facing: tgen_port4..tgen_port7 (dest ranks)

    Traffic pattern:
      All 4 source ranks send to all 4 dest ranks simultaneously.
      Leaf0 → Spine → Leaf1 path is traversed for every flow.
      With ECMP, Spine should distribute flows across its uplinks.

    Expected result:
      All 4 dest ports receive traffic (all ECMP paths used).
      Per-path Rx imbalance < 30% (fair distribution).

    STC API note: multi-DUT fixture path relies on fanout_graph_facts_multidut
    and duthosts list ordering matching port_ids order from get_snappi_ports.
    Confirm multi-DUT conftest behavior when stc-x2rest Topo2 support lands
    (CIPENH-1476 Topo2 track, ETA TBD after Aug-8 first version).

    Key differentiator vs Ixia: vendor-neutral helper (no api._ixnetwork).

    References:
      Test plan: sonic-net/sonic-mgmt#21514
      TC1 reference: sonic-net/sonic-mgmt#23240
    """
    snappi_port_list = get_snappi_ports
    pytest_require(len(snappi_port_list) >= 8,
                   "TC7 Clos topology needs 8 tgen ports (4 Leaf0 + 4 Leaf1 server-facing)")

    tconfig, plist, sports = snappi_dut_base_config(duthosts, snappi_port_list, snappi_api)
    snappi_dut_port_map = snappi_dut_port_mapping(sports)
    port_ids = [pc.id for pc in plist]

    priority_to_dscp = derive_priority_to_dscp(prio_dscp_map)
    queue = 3
    dscp = priority_to_dscp[queue]
    qids = [f"UC{queue}"]

    def conn(ack_nak_dscp):
        return {
            "choice": "reliable_connection",
            "reliable_connection": {
                "ack": {"ip_dscp": ack_nak_dscp, "ecn_value": "non_ect"},
                "nak": {"ip_dscp": ack_nak_dscp, "ecn_value": "non_ect"},
                "enable_retransmission_timeout": True,
                "retransmission_timeout_value": 5000,
                "retransmission_retry_count": 254,
            },
        }

    def lossless_cfg_fn(q, message_size_kb):
        d = priority_to_dscp[q]
        return {
            "mtu": 9100,
            "qp_configs": [{"message_size_unit": "kb", "message_size": message_size_kb,
                             "dscp": d}],
            "cnp": {"ip_dscp": CNP_DSCP, "ecn_value": "non_ect"},
            "dcqcn_settings": {"enable_dcqcn": False},
            "connection_type": conn(d),
            "rocev2_port_config": {"transmit_type": "target_line_rate",
                                   "target_line_rate": TX_RATE_PERCENT},
        }

    lossless_spec = [(queue, 128)] * 4  # same config for all 4 source ranks
    topology, src_port_ids, dst_port_ids = build_clos_ecmp_topology(
        port_ids, lossless_cfg_fn, lossless_spec)
    logger.info(f"TC7 Clos topology: src={src_port_ids} dst={dst_port_ids}")

    _merged_df, flow_df, dut_queue_df, _sched_df = collect_flow_queue_stats_portable(
        snappi_api=snappi_api, duthosts=duthosts, plist=plist, tconfig=tconfig,
        snappi_dut_port_map=snappi_dut_port_map, topology=topology,
        prio_dscp_map=prio_dscp_map, queue_ids=qids,
        traffic_duration=TRAFFIC_DURATION, sample_rates=True)

    # ECMP distribution check
    ecmp_hash_verify(flow_df, dst_port_ids, imbalance_threshold=0.30)

    # Base lossless checks
    checks = [
        make_check("message_fail != 0 or frame_delta != 0",
                ["flow_name", "port_tx", "port_rx", "ip_dscp", "message_fail", "frame_delta"],
                "All messages complete, no loss", "message_fail=0 & frame_delta=0 required"),
        make_check("nak_tx != 0 or nak_rx != 0 or frame_sequence_error != 0",
                ["flow_name", "port_tx", "port_rx", "ip_dscp", "nak_tx", "nak_rx", "frame_sequence_error"],
                "No NAK/sequence errors", "nak_tx=0 & nak_rx=0 & frame_sequence_error=0 required"),
    ]
    assert_queries(flow_df, checks)
    logger.info("*** TC7 PASSED ***")
