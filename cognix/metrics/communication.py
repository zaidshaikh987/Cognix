def message_reduction_ratio(actual_messages: int, max_possible_messages: int) -> float:
    if max_possible_messages == 0:
        return 0.0
    return 1.0 - (actual_messages / max_possible_messages)

def communication_overhead(messages: int, payload_size_bytes: int) -> dict:
    return {
        "total_messages": messages,
        "total_payload_bytes": payload_size_bytes,
        "average_payload_bytes": payload_size_bytes / messages if messages > 0 else 0.0
    }

def information_efficiency(uncertainty_reduction: float, messages_used: int) -> float:
    if messages_used == 0:
        return 0.0
    return uncertainty_reduction / messages_used
