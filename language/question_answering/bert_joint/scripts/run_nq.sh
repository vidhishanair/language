OUTPUT=gs://fat_storage/baseline_seq512_unk0.1/output_lr3e-5.epochs1.seed1
BERT_BASE_DIR=gs://fat_storage/pretrained_bert//wwm_uncased_L-24_H-1024_A-16

python -m language.question_answering.bert_joint.run_nq \
  --logtostderr \
  --bert_config_file=$BERT_BASE_DIR/bert_config.json \
  --vocab_file=gs://fat_storage/bert-joint-baseline/vocab-nq.txt \
  --train_precomputed_file=gs://fat_storage/baseline_seq512_unk0.1/train/nq-train-*.tfrecords \
  --predict_file=/home/vbalacha/datasets/v1.0/dev/*.jsonl.gz \
  --train_num_precomputed=1171257 \
  --learning_rate=3e-5 \
  --num_train_epochs=1 \
  --seed=1 \
  --max_seq_length=512 \
  --save_checkpoints_steps=5000 \
  --init_checkpoint=$BERT_BASE_DIR/bert_model.ckpt \
  --do_train \
  --do_predict \
  --output_prediction_file=$OUTPUT/predictions.json \
  --output_dir=$OUTPUT \
   --use_tpu=True \
  --tpu_name=$TPU_NAME

mkdir tmp

gsutil cp $OUTPUT/predictions.json tmp/

python -m natural_questions.nq_eval --logtostderr --gold_path=/home/vbalacha/datasets/v1.0/dev/nq-dev-0?.jsonl.gz --predictions_path=tmp/predictions.json > tmp/metrics.json

cat tmp/metrics.json

gsutil cp tmp/metrics.json $OUTPUT/

rm -rf tmp
