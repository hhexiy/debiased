#sbatch --job-name kde --export=command="make train-embed task=MNLI from=output/mnli/bert-base/36ad8f42-d886-11e9-96a3-246e96dd6c80 exp=kde/mnli/bert-base" batch/run_test.sh

#sbatch --job-name kde --export=command="make train-embed task=MNLI from=output/mnli/roberta/25338112-e471-11e9-99ea-b496910c63bc exp=kde/mnli/roberta-base" batch/run_test.sh

#sbatch --job-name kde --export=command="make train-embed task=QQP-wang from=output/qqp/bert-base/12b8c4ee-ee16-11e9-a882-246e96dd351e exp=kde/qqp/bert-base" batch/run_test.sh

#sbatch --job-name kde --export=command="make train-embed task=QQP-wang from=output/qqp/roberta-base/0b843f66-ee1e-11e9-9d59-b496910c6a64 exp=kde/qqp/roberta-base" batch/run_test.sh

sbatch --job-name kde --export=command="make train-init-embed task=MNLI from=output/mnli/bert-base/36ad8f42-d886-11e9-96a3-246e96dd6c80 exp=kde/mnli/bert-base-init" batch/run_test.sh

sbatch --job-name kde --export=command="make train-init-embed task=MNLI from=output/mnli/roberta/25338112-e471-11e9-99ea-b496910c63bc exp=kde/mnli/roberta-base-init" batch/run_test.sh

sbatch --job-name kde --export=command="make train-init-embed task=QQP-wang from=output/qqp/bert-base/12b8c4ee-ee16-11e9-a882-246e96dd351e exp=kde/qqp/bert-base-init" batch/run_test.sh

sbatch --job-name kde --export=command="make train-init-embed task=QQP-wang from=output/qqp/roberta-base/0b843f66-ee1e-11e9-9d59-b496910c6a64 exp=kde/qqp/roberta-base-init" batch/run_test.sh
