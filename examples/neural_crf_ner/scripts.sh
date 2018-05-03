
python main.py -mode=prepro | tee logs/prepro.log

python main.py -model_type=rnn -mode=train -model_name=rnn_rerun -num_epochs=15 | tee logs/rnn_rerun.log
python main.py -model_type=birnn -mode=train -model_name=birnn_rerun | tee logs/birnn_rerun.log
python main.py -model_type=birnn_crf -mode=train -model_name=birnn_crf -num_epochs=10 -save_epoch_freq=1 -batch_size=1 | tee logs/birnn_crf.log
python main.py -model_type=birnn_crf -mode=train -model_name=birnn_crf_rerun -num_epochs=10 -save_epoch_freq=1 -batch_size=1 | tee logs/birnn_crf_rerun.log

python main.py -model_type=birnn_crf -mode=train -model_name=birnn_crf_rerun -num_epochs=6 -save_epoch_freq=1 -batch_size=1 -emb_size=96 -hidden_size=96  | tee logs/birnn_crf_e96_h96.log
python main.py -model_type=birnn_crf -mode=train -model_name=birnn_crf_rerun -num_epochs=6 -save_epoch_freq=1 -batch_size=1 -emb_size=192 -hidden_size=96  | tee logs/birnn_crf_e192_h96.log
python main.py -model_type=birnn_crf -mode=train -model_name=birnn_crf_rerun -num_epochs=6 -save_epoch_freq=1 -batch_size=1 -emb_size=192 -hidden_size=128  | tee logs/birnn_crf_e192_h128.log

