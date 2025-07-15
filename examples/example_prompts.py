fg_prompt = "a 20-year-old male Asian student, side part hair, wearing a red and black checkered shirt, a white undershirt, and blue jeans,"

bg_act_prompts = [
    dict(
        bg = "In an empty background,",
        act = [
            "standing, hands hung naturally at his sides, with a blank expression."
        ],
    ),
    dict(
        bg = "A bright office, a computer desk, a window in the background,",
        act = [
            "sitting, looking at the computer, with one hand resting on their forehead, filling exhaustion, side view",
            "smiling, sitting, side view",
        ],
    ),
    dict(
        bg = "outside the entrance of a large building,",
        act = [
            "standing, looking at the phone on the hands, displaying a melancholic expression",
            "standing with a blank expression, raising one fist to encourage, standing tall with their head held high, side view",
            "jumping up to the air, waving both fists, laughing out",
        ],
        height=1280,
        width=720,
    ),
    dict(
        bg = "A bright auditorium, with a large screen in the background,",
        act = [
            "standing on stage, making a speech"
        ],
    ),
    dict(
        bg = "In a sunny beach, with waves in the background,",
        act = [
            "lying on the beach, under a beach umbrella, laughing, wearing sunglasses"
        ]
    ),
    dict(
        bg = "In a bustling airport,",
        act = [
            "walking, carrying a suitcase, back to the camera, back view"
        ]
    ),
    dict(
        bg = "In a dimly lit bedroom with a bed, a small desk lamp and book shelf in the background",
        act = [
            "sitting on the bed, rubbing his eyes, looking down",
            "lying flat on his back, falling in sleep, closing eyes, head resting on the pillow",
        ],
        height=720,
        width=1280,
    )
]
