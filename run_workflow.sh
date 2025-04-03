#!/bin/bash

set -e
set -o pipefail

# Default steps
RUN_DOWNLOAD=true
RUN_CAPTIONS=true
RUN_TRAIN=true
RUN_EVAL=false

# Parse flags
for arg in "$@"; do
    case $arg in
        --no-download) RUN_DOWNLOAD=false ;;
        --no-captions) RUN_CAPTIONS=false ;;
        --no-train)    RUN_TRAIN=false ;;
        --evaluate)    RUN_EVAL=true ;;
        *)
            echo "⚠️ Unknown option: $arg"
            exit 1
            ;;
    esac
done

if $RUN_DOWNLOAD; then
    echo "🔽 Step 1: Downloading dataset (if needed)..."
    python -m scripts.download_data || echo "✅ Dataset already exists, skipping."
else
    echo "⏭️ Skipping dataset download."
fi

if $RUN_CAPTIONS; then
    echo "🧹 Step 2: Generating rule-based captions..."
    python -m scripts.preprocess_and_caption
else
    echo "⏭️ Skipping caption generation."
fi

if $RUN_TRAIN; then
    echo "🏋️ Step 3: Training the model..."
    python -m scripts.train
else
    echo "⏭️ Skipping model training."
fi

echo "🖼️ Step 4: Predicting a sample image..."
SAMPLE_IMAGE=$(find data/raw/Coronahack-Chest-XRay-Dataset/Coronahack-Chest-XRay-Dataset/train -type f -name "*.jpeg" | head -n 1)

if [ -z "$SAMPLE_IMAGE" ]; then
    echo "❌ No image found for prediction test."
else
    echo "Using sample image: $SAMPLE_IMAGE"
    python -m scripts.predict --image_path "$SAMPLE_IMAGE"
fi

if $RUN_EVAL; then
    echo "📊 Step 5: Evaluating model on test set..."
    python -m scripts.evaluate
else
    echo "⏭️ Skipping full evaluation."
fi