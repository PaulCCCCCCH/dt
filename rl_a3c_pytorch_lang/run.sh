python main.py --env PongDeterministic-v4 --workers 8 --amsgrad True \
--lstm-size 100 \
--emb-dim 25 \
--emb-to-load 0 \
--load \
--render \
--use-language \
--emb-path ../emb/glove_twitter_25d_changed.txt \

# --lm-dir ./pre_trained_lang_model_25d \

# --use-full-emb
# --alpha-mode none
