for i in `echo {01..10}`; do
echo "mcmc:
    seed: `echo $RANDOM` 
    num_steps: 1000000
    num_chains: 10
    num_pairs: 10
    error: 0.1
    do_exchange: True
    dump_all: True
    suffix: pt_1M
    dir_name: m_a6/run_${i}
    initial_variance: [0.0001, 0.0001, 0.0001, 0.0001, 0.0001]
model:
    class_file: models
    object: models.PapaModelA6()
    param_names: [alpha, D, Co, Ns, K, K1]
    initial_values: 
    prior_min: [0.0001, 0.0001, 0.0001, 1., 3.5, 3.5]
    prior_max: [1., 2., 2., 200., 9., 9.]" > configs/config_a6/config_a6_${i}.yml
done

for i in `echo {01..10}`; do
	qsubHead.sh -s run_ma6_${i} -w 500 -m 5gb
	echo python mcmc.py configs/config_a6/config_a6_${i}.yml >> run_ma6_${i}.qsub
done

#############################################################################################

for  ii in `echo {01..10}`; do
    echo "mcmc:
    seed: `echo $RANDOM`
    num_steps: 1000000
    num_chains: 10
    num_pairs: 10
    error: 0.1
    do_exchange: True
    dump_all: True
    suffix: pt_1M
    dir_name: m_b7/run_${ii}
    initial_variance: [0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001]
model:
    class_file: models
    object: models.PapaModelB7()
    param_names: [alpha, D, Co, Ns, K, K1, K2]
    initial_values: 
    prior_min: [0.0001, 0.0001, 0.0001, 1., 3.5, 3.5, 3.5]
    prior_max: [1., 2., 2., 200., 9., 9., 9.]" > configs/config_b7/config_b7_${ii}.yml
done

for ii in `echo {01..10}`; do
    qsubHead.sh -s run_mb7_${ii} -w 500 -m 5gb
    echo python mcmc.py configs/config_b7/config_b7_${ii}.yml >> run_mb7_${ii}.qsub
done

#############################################################################################

for i in `echo {01..10}`; do
echo "mcmc:
    seed: `echo $RANDOM`
    num_steps: 1000000
    num_chains: 10
    num_pairs: 10
    error: 0.1
    do_exchange: True
    dump_all: True
    suffix: pt_1M
    dir_name: m_b7r/run_${i}
    initial_variance: [0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001]
model:
    class_file: models
    object: models.PapaModelB7r()
    param_names: [alpha, D, Co, Ns, K, K1, K3]
    initial_values: 
    prior_min: [0.0001, 0.0001, 0.0001, 1., 3.5, 3.5, 3.5]
    prior_max: [1., 2., 2., 200., 9., 9., 9.]" > configs/config_b7r/config_b7r_${i}.yml
done

for i in `echo {01..10}`; do
	qsubHead.sh -s run_mb7r_${i} -w 500 -m 5gb
	echo python mcmc.py configs/config_b7r/config_b7r_${i}.yml >> run_mb7r_${i}.qsub
done

#############################################################################################

for ii in `echo {01..10}`; do
echo "mcmc:
    seed: `echo $RANDOM`
    num_steps: 1000000
    num_chains: 10
    num_pairs: 10
    error: 0.1
    do_exchange: True
    dump_all: True
    suffix: pt_1M
    dir_name: m_c8/run_${ii}
    initial_variance: [0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001]
model:
    class_file: models
    object: models.PapaModelC8()
    param_names: [alpha, D, Co, Ns, K, K1, K2, K3]
    initial_values: 
    prior_min: [0.0001, 0.0001, 0.0001, 1., 3.5, 3.5, 3.5, 3.5]
    prior_max: [1., 2., 2., 200., 9., 9., 9., 9.]" > configs/config_c8/config_c8_${ii}.yml
done

for i in `echo {01..10}`; do
	qsubHead.sh -s run_mc8_${i} -w 500 -m 5gb
	echo python mcmc.py configs/config_c8/config_c8_${i}.yml >> run_mc8_${i}.qsub
done

#############################################################################################

for i in `echo {01..10}`; do
echo "mcmc:
    seed: `echo $RANDOM`
    num_steps: 1000000
    num_chains: 10
    num_pairs: 10
    error: 0.1
    do_exchange: True
    dump_all: True
    suffix: pt_1M
    dir_name: m_B_Kr7/run_${i}
    initial_variance: [0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001]
model:
    class_file: models
    object: models.PapaModel_B_Kr7()
    param_names: [alpha, D, Co, Ns, K, K1, K2]
    initial_values: 
    prior_min: [0.0001, 0.0001, 0.0001, 1., 3.5, 3.5, 3.5]
    prior_max: [1., 2., 2., 200., 9., 9., 9.]" > configs/config_B_Kr7/config_B_Kr7_${i}.yml
done

for i in `echo {01..10}`; do
	qsubHead.sh -s run_mB_Kr7_${i} -w 500 -m 5gb
	echo python mcmc.py configs/config_B_Kr7/config_B_Kr7_${i}.yml >> run_mB_Kr7_${i}.qsub
done

#############################################################################################

for i in `echo {01..10}`; do
echo "mcmc:
    seed: `echo $RANDOM`
    num_steps: 1000000
    num_chains: 10
    num_pairs: 10
    error: 0.1
    do_exchange: True
    dump_all: True
    suffix: pt_1M
    dir_name: m_B_Kr7r/run_${i}
    initial_variance: [0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001]
model:
    class_file: models
    object: models.PapaModel_B_Kr7r()
    param_names: [alpha, D, Co, Ns, K, K1, K3]
    initial_values: 
    prior_min: [0.0001, 0.0001, 0.0001, 1., 3.5, 3.5, 3.5]
    prior_max: [1., 2., 2., 200., 9., 9., 9.]" > configs/config_B_Kr7r/config_B_Kr7r_${i}.yml
done

for i in `echo {01..10}`; do
    qsubHead.sh -s run_mB_Kr7r_${i} -w 500 -m 5gb
    echo python mcmc.py configs/config_B_Kr7r/config_B_Kr7r_${i}.yml >> run_mB_Kr7r_${i}.qsub
done

#############################################################################################

for i in `echo {01..10}`; do
echo "mcmc:
    seed: `echo $RANDOM`
    num_steps: 1000000
    num_chains: 10
    num_pairs: 10
    error: 0.1
    do_exchange: True
    dump_all: True
    suffix: pt_1M
    dir_name: m_B_Kr8/run_${i}
    initial_variance: [0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001]
model:
    class_file: models
    object: models.PapaModel_B_Kr8()
    param_names: [alpha, D, Co, Ns, K, K1, K2, K3]
    initial_values: 
    prior_min: [0.0001, 0.0001, 0.0001, 1., 3.5, 3.5, 3.5, 3.5]
    prior_max: [1., 2., 2., 200., 9., 9., 9., 9.]" > configs/config_B_Kr8/config_B_Kr8_${i}.yml
done

for i in `echo {01..10}`; do
	qsubHead.sh -s run_mB_Kr8_${i} -w 500 -m 5gb
	echo python mcmc.py configs/config_B_Kr8/config_B_Kr8_${i}.yml >> run_mB_Kr8_${i}.qsub
done
