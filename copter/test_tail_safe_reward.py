from structures import calculate_tail_safe_reward


CALIBRATION = {
    "mixed": {
        "aggressive": (0.4176923, 0.0095331, 0.0330344, 0.0136223, 0.0348698),
        "balanced": (0.4186569, 0.0101109, 0.0345901, 0.0114572, 0.0262791),
        "permissive": (0.4282938, 0.0128165, 0.0404799, 0.0090820, 0.0177290),
    },
    "incast": {
        "aggressive": (0.1707558, 0.0612948, 0.0966416, 0.0899403, 0.1129346),
        "balanced": (0.1730417, 0.0581588, 0.0929619, 0.0861461, 0.1079858),
        "permissive": (0.1810733, 0.0657608, 0.1038240, 0.0865709, 0.1110583),
    },
}


def test_calibration_ordering():
    rewards = {
        task: {
            action: calculate_tail_safe_reward(*values)["reward"]
            for action, values in actions.items()
        }
        for task, actions in CALIBRATION.items()
    }
    assert max(rewards["mixed"], key=rewards["mixed"].get) == "permissive"
    assert max(rewards["incast"], key=rewards["incast"].get) == "balanced"


def test_reward_is_bounded():
    assert calculate_tail_safe_reward(0, 1, 1, 1, 1)["reward"] == -1.0
    assert calculate_tail_safe_reward(1, 0, 0, 0, 0)["reward"] == 1.0


if __name__ == "__main__":
    test_calibration_ordering()
    test_reward_is_bounded()
    print("tail-safe reward tests passed")
