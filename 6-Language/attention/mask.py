import sys
import tensorflow as tf

from PIL import Image, ImageDraw, ImageFont
from transformers import AutoTokenizer, TFBertForMaskedLM

# Pre-trained masked language model
MODEL = "bert-base-uncased"

# Number of predictions to generate
K = 3

# Constants for generating attention diagrams
FONT = ImageFont.truetype("assets/fonts/OpenSans-Regular.ttf", 28)
GRID_SIZE = 40
PIXELS_PER_WORD = 200


def main():
    text = input("Text: ")

    # Tokenize input
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    inputs = tokenizer(text, return_tensors="tf")
    mask_token_index = get_mask_token_index(tokenizer.mask_token_id, inputs)
    if mask_token_index is None:
        sys.exit(f"Input must include mask token {tokenizer.mask_token}.")

    # Use model to process input
    model = TFBertForMaskedLM.from_pretrained(MODEL)
    result = model(**inputs, output_attentions=True)

    # Generate predictions
    mask_token_logits = result.logits[0, mask_token_index]
    top_tokens = tf.math.top_k(mask_token_logits, K).indices.numpy()
    for token in top_tokens:
        print(text.replace(tokenizer.mask_token, tokenizer.decode([token])))

    # Visualize attentions
    visualize_attentions(inputs.tokens(), result.attentions)


def get_mask_token_index(mask_token_id, inputs):
    """
    Return the index of the token with the specified `mask_token_id`, or
    `None` if not present in the `inputs`.
    """
    # inputs is a transformers.BatchEncoding
    # The token ids are available under inputs["input_ids"].
    # Try to obtain a plain Python list of the tokens for the first (and only) item.
    try:
        # If it's a TF tensor: inputs["input_ids"][0].numpy()
        seq = inputs["input_ids"][0].numpy().tolist()
    except Exception:
        # Otherwise, try to convert to list directly (covers list of lists)
        try:
            seq = list(inputs["input_ids"][0])
        except Exception:
            # If all else fails, return None
            return None

    for i, token_id in enumerate(seq):
        # token_id might be a numpy/int, so coerce to int for comparison
        try:
            if int(token_id) == int(mask_token_id):
                return i
        except Exception:
            continue

    return None


def get_color_for_attention_score(attention_score):
    """
    Return a tuple of three integers representing a shade of gray for the
    given `attention_score`. Each value should be in the range [0, 255].
    """
    # Clamp attention_score into [0, 1]
    try:
        score = float(attention_score)
    except Exception:
        score = 0.0
    if score < 0:
        score = 0.0
    if score > 1:
        score = 1.0

    # Scale to [0, 255] and round to nearest int
    gray = int(round(score * 255.0))

    # Ensure range safety
    if gray < 0:
        gray = 0
    if gray > 255:
        gray = 255

    return (gray, gray, gray)


def visualize_attentions(tokens, attentions):
    """
    Produce a graphical representation of self-attention scores.

    For each attention layer, one diagram should be generated for each
    attention head in the layer. Each diagram should include the list of
    `tokens` in the sentence. The filename for each diagram should
    include both the layer number (starting count from 1) and head number
    (starting count from 1).
    """
    # attentions is a tuple/list of tensors with shape (batch, num_heads, seq_len, seq_len)
    # We assume batch dimension is 1 and beam index is 0 as mentioned in spec.
    num_layers = len(attentions)

    for layer_idx in range(num_layers):
        # attentions[layer_idx] has shape (batch, num_heads, seq_len, seq_len)
        layer = attentions[layer_idx]
        # get batch=0
        try:
            batch0 = layer[0]
        except Exception:
            # If indexing fails, skip layer
            continue

        # number of heads in this layer
        try:
            num_heads = int(batch0.shape[0])
        except Exception:
            # fallback to len(...)
            try:
                num_heads = len(batch0)
            except Exception:
                continue

        for head_idx in range(num_heads):
            # get attention weights for this head: shape (seq_len, seq_len)
            try:
                attention_weights = batch0[head_idx]
                # Convert tf.Tensor to numpy array if needed
                if hasattr(attention_weights, "numpy"):
                    attention_weights = attention_weights.numpy()
            except Exception:
                # If something goes wrong, skip this head
                continue

            # layer_number and head_number should be 1-indexed for filenames
            generate_diagram(layer_idx + 1, head_idx + 1,
                             tokens, attention_weights)


def generate_diagram(layer_number, head_number, tokens, attention_weights):
    """
    Generate a diagram representing the self-attention scores for a single
    attention head. The diagram shows one row and column for each of the
    `tokens`, and cells are shaded based on `attention_weights`, with lighter
    cells corresponding to higher attention scores.

    The diagram is saved with a filename that includes both the `layer_number`
    and `head_number`.
    """
    # Create new image
    image_size = GRID_SIZE * len(tokens) + PIXELS_PER_WORD
    img = Image.new("RGBA", (image_size, image_size), "black")
    draw = ImageDraw.Draw(img)

    # Draw each token onto the image
    for i, token in enumerate(tokens):
        # Draw token columns
        token_image = Image.new("RGBA", (image_size, image_size), (0, 0, 0, 0))
        token_draw = ImageDraw.Draw(token_image)
        token_draw.text(
            (image_size - PIXELS_PER_WORD, PIXELS_PER_WORD + i * GRID_SIZE),
            token,
            fill="white",
            font=FONT
        )
        token_image = token_image.rotate(90)
        img.paste(token_image, mask=token_image)

        # Draw token rows
        _, _, width, _ = draw.textbbox((0, 0), token, font=FONT)
        draw.text(
            (PIXELS_PER_WORD - width, PIXELS_PER_WORD + i * GRID_SIZE),
            token,
            fill="white",
            font=FONT
        )

    # Draw each word
    for i in range(len(tokens)):
        y = PIXELS_PER_WORD + i * GRID_SIZE
        for j in range(len(tokens)):
            x = PIXELS_PER_WORD + j * GRID_SIZE
            color = get_color_for_attention_score(attention_weights[i][j])
            draw.rectangle((x, y, x + GRID_SIZE, y + GRID_SIZE), fill=color)

    # Save image
    img.save(f"Attention_Layer{layer_number}_Head{head_number}.png")


if __name__ == "__main__":
    main()
