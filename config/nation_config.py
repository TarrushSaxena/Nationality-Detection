# Branching Logic Configuration

# Nationality Constants
INDIAN = "Indian"
US = "United States"
AFRICAN = "African"
OTHER = "Other"

# Branching Map
BRANCHES = {
    INDIAN: ["Emotion", "Age", "Dress Color"],
    US: ["Emotion", "Age", "Dress Color"],
    AFRICAN: ["Emotion", "Age", "Dress Color"],
    OTHER: ["Emotion", "Age", "Dress Color"]
}

# Color Detection
COLOR_K = 3  # Number of clusters for K-Means
TORSO_OFFSET_FACTOR = 1.5  # How much to extend face box downwards

# Model Paths
MODEL_PATHS = {
    "nationality": "models/nation_model_best.h5",
    "emotion": "models/emotion_model_best.h5",
    "age": "models/age_model_best.h5"
}
