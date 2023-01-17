#!/usr/bin/env zsh


target_dir=$1

function error
{
    if [ -z "$1" ]
    then
        message="fatal error"
    else
        message="fatal error: $1"
    fi

    echo $message
    echo "finished at $(date)"
    exit 1
}

stage=0
stop_stage=100

# Create dict.txt and words.txt
if [ $stage -le 0 ] && [ $stop_stage -ge 0 ]; then
    cat $target_dir/train.wrd $target_dir/valid.wrd $target_dir/test.wrd > $target_dir/all.wrd
    cat $target_dir/train.phn $target_dir/valid.phn $target_dir/test.phn > $target_dir/all.phn

    python $FAIRSEQ_ROOT/fairseq_cli/preprocess.py --dataset-impl mmap --trainpref $target_dir/all.wrd --only-source --destdir $target_dir --thresholdsrc 0 --padding-factor 1 --dict-only
    cut -f1 -d' ' $target_dir/dict.txt | grep -v -x '[[:punct:]]*' | grep -Pv '\d\d\d\d\d+' >! $target_dir/words.txt
fi

if [ $stage -le 1 ] && [ $stop_stage -ge 1 ]; then
    cp $target_dir/all.wrd $target_dir/lm.upper.lid.txt
    python scripts/g2p_wrd_to_phn.py \
            --wrd_path $target_dir/all.wrd \
            --in_path $target_dir/words.txt \
            --out_path $target_dir/phones.txt

    for split in train valid; do
        for suffix in phn wrd; do 
            cp $target_dir/${split}.${suffix} $target_dir/feat
            cp $target_dir/${split}.${suffix} $target_dir/feat/precompute_pca512
            cp $target_dir/${split}.${suffix} $target_dir/feat/precompute_pca512_cls128_mean
            cp $target_dir/${split}.${suffix} $target_dir/feat/precompute_pca512_cls128_mean_pooled
        done
    done
fi

if [ $stage -le 2 ] && [ $stop_stage -ge 2 ]; then
    paste $target_dir/words.txt $target_dir/phones.txt >! $target_dir/lexicon.lst

    python $FAIRSEQ_ROOT/fairseq_cli/preprocess.py --dataset-impl mmap --trainpref $target_dir/phones.txt --only-source --destdir $target_dir/phones --thresholdsrc 1 --padding-factor 1 --dict-only

    python $FAIRSEQ_ROOT/examples/wav2vec/unsupervised/scripts/filter_lexicon.py -d $target_dir/phones/dict.txt < $target_dir/lexicon.lst >! $target_dir/lexicon_filtered.lst
    python $FAIRSEQ_ROOT/examples/wav2vec/unsupervised/scripts/phonemize_with_sil.py -s 0.25 --surround --lexicon $target_dir/lexicon_filtered.lst < $target_dir/lm.upper.lid.txt >! $target_dir/phones/lm.phones.filtered.txt
    cp $target_dir/phones/dict.txt $target_dir/phones/dict.phn.txt
    echo "<SIL> 0" >> $target_dir/phones/dict.phn.txt
    python $FAIRSEQ_ROOT/fairseq_cli/preprocess.py --dataset-impl mmap --trainpref $target_dir/phones/lm.phones.filtered.txt --workers 70 --only-source --destdir $target_dir/phones --srcdict $target_dir/phones/dict.phn.txt

    $KENLM_ROOT/lmplz -o 4 < $target_dir/lm.upper.lid.txt --discount_fallback --prune 0 0 0 3 >! $target_dir/kenlm.wrd.o40003.arpa
    $KENLM_ROOT/build_binary $target_dir/kenlm.wrd.o40003.arpa $target_dir/kenlm.wrd.o40003.bin
fi

if [ $stage -le 3 ] && [ $stop_stage -ge 3 ]; then
    lg=$lg python $FAIRSEQ_ROOT/examples/speech_recognition/kaldi/kaldi_initializer.py kaldi_root=$KALDI_ROOT fst_dir=$target_dir/fst/phn_to_words_sil lm_arpa=$target_dir/kenlm.wrd.o40003.arpa wav2letter_lexicon=$target_dir/lexicon_filtered.lst data_dir=$target_dir/phones in_labels=phn "blank_symbol='<SIL>'"
    lg=$lg python $FAIRSEQ_ROOT/examples/speech_recognition/kaldi/kaldi_initializer.py kaldi_root=$KALDI_ROOT fst_dir=$target_dir/fst/phn_to_words lm_arpa=$target_dir/kenlm.wrd.o40003.arpa wav2letter_lexicon=$target_dir/lexicon_filtered.lst data_dir=$target_dir/phones in_labels=phn

    $KENLM_ROOT/lmplz -o 4 < $target_dir/phones/lm.phones.filtered.txt --discount_fallback >! $target_dir/phones/lm.phones.filtered.04.arpa
    $KENLM_ROOT/build_binary $target_dir/phones/lm.phones.filtered.04.arpa $target_dir/phones/lm.phones.filtered.04.bin
    $KENLM_ROOT/lmplz -o 6 < $target_dir/phones/lm.phones.filtered.txt --discount_fallback >! $target_dir/phones/lm.phones.filtered.06.arpa
    $KENLM_ROOT/build_binary $target_dir/phones/lm.phones.filtered.06.arpa $target_dir/phones/lm.phones.filtered.06.bin

    lg=$lg python $FAIRSEQ_ROOT/examples/speech_recognition/kaldi/kaldi_initializer.py kaldi_root=$KALDI_ROOT fst_dir=$target_dir/fst/phn_to_phn_sil lm_arpa=$target_dir/phones/lm.phones.filtered.06.arpa data_dir=$target_dir/phones in_labels=phn "blank_symbol='<SIL>'"
fi