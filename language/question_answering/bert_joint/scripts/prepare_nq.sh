for j in {00..49}
do
        echo $j
        python -m language.question_answering.bert_joint.prepare_nq_data \
                --logtostderr \
                --input_jsonl /home/vbalacha/datasets/v1.0/train/nq-train-$j.jsonl.gz \
                --output_tfrecord gs://fat_storage/baseline_seq512_unk0.1/train/nq-train-$j.tfrecords \
                --max_seq_length=512 \
                --include_unknowns=0.1 \
                --vocab_file=gs://fat_storage/bert-joint-baseline/vocab-nq.txt > log/train_$j.log 2>&1 &
done
wait
echo "All Done"