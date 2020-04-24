python main.py --env PongDeterministic-v4 --workers 1 --amsgrad True \
--emb-path ../emb/glove_twitter_25d_changed.txt \
--lstm-size 100 \
--emb-dim 25 \
--load \
--use-language \
--render \
--emb-to-load 300
# --use-full-emb
