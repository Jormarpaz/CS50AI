# Analysis

## Layer 2, Head 5 — Determiners to their Nouns

This attention head often pays attention from determiners (tokens like "the", "a", "an", "my") to the head noun that the determiner modifies. In multiple examples, when a determiner appears, the corresponding row in this head's attention heatmap shows higher weights towards the noun immediately following the determiner (or the noun that is the head of the noun phrase).

Example Sentences:

- "the red door was on the palm of the hand"
- "a dreadful mess arrived on thursday"

(Procedure: run the model on each sentence with `mask.py`, open the attention image for Layer 2 Head 5 and look at the row corresponding to the determiner token — it often lights up the noun token.)

## Layer 7, Head 3 — Pronouns attending to antecedent nouns

This attention head often shows higher attention from personal pronouns (like "he", "she", "him", "her", "it") to the noun or proper noun earlier in the sentence that is likely to be the antecedent. In our example sentences, the pronoun's row has stronger attention weights toward the candidate antecedent token(s) earlier in the sequence.

Example Sentences:

- "holmes smiled and he sat on the armchair"
- "the companion chuckled because she had a word with him"

(Procedure: run the model on each sentence and inspect Layer 7 Head 3's diagram — check the row for the pronoun and see whether cells corresponding to the antecedent nouns are brighter.)
