
for numbersnps in 16 64 128 256 512;
do
    for num_batches in 2 5 50;
    do

        python DifferentialPrivacyTest.py -i synSNPs_dataset_shuffled/GenoType123_top_${numbersnps}SNPs_DM_shuffle_ --output synSNPs${numbersnps}_randomBatch_${num_batches}Batches --num_batches ${num_batches}

    done
done


