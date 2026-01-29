import datetime
import os
import pprint
import time
import threading
import torch as th
import numpy as np
from types import SimpleNamespace as SN
from utils.logging import Logger
from utils.timehelper import time_left, time_str
from os.path import dirname, abspath

from learners import REGISTRY as le_REGISTRY
from runners import REGISTRY as r_REGISTRY
from controllers import REGISTRY as mac_REGISTRY
from components.episode_buffer import ReplayBuffer
from components.transforms import OneHot

import socket
import wandb
import logging

from modules.step.models import Qdifference_Transformer
from modules.step.models import Planning_Transformer
from modules.step.models import ObsPredictorMLP
from Attacker_learners.RL_Based.archive import Archive
from Attacker_learners.RL_Based.population import Population

def run(_run, _config, _log):
    logging.getLogger("requests").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    
    # check args sanity
    _config = args_sanity_check(_config, _log)

    args = SN(**_config)
    args.device = "cuda" if args.use_cuda else "cpu"

    # setup loggers
    logger = Logger(_log)

    _log.info("Experiment Parameters:")
    experiment_params = pprint.pformat(_config,
                                       indent=4,
                                       width=1)
    _log.info("\n\n" + experiment_params + "\n")

    # configure tensorboard logger
    # Get map name from environment arguments
    map_name = args.env_args.get("map_name", "unknown_map")
    unique_token = "{}_{}_{}_".format(args.name, map_name, datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
    args.unique_token = unique_token
    if args.use_tensorboard:
        tb_logs_direc = os.path.join(dirname(dirname(abspath(__file__))), "results", "tb_logs")
        tb_exp_direc = os.path.join(tb_logs_direc, "{}").format(unique_token)
        logger.setup_tb(tb_exp_direc)
    if args.use_wandb:
        wandb_logs_direc = os.path.join(dirname(dirname(dirname(abspath(__file__)))), "results", "wandb")
        wandb_exp_direc = os.path.join(wandb_logs_direc, "{}").format(unique_token)

        key=""    # key
        wandb.login(key=key)
        
        # Get algorithm name from config
        alg_name = args.name
        
        run = wandb.init(config=args,
                        project=map_name,
                        entity='',
                        notes=socket.gethostname(),
                        name=alg_name,
                        group='',
                        dir=str(wandb_exp_direc),
                        job_type="",
                        mode="disabled",
                        reinit=True)

    # sacred is on by default
    logger.setup_sacred(_run)

    # Run and train
    if args.attacker_type =="RL":
        run_sequential_RL(args=args, logger=logger)
    elif args.attacker_type =="rule":
        run_sequential_rule(args=args, logger=logger)
    # Clean up after finishing
    print("Exiting Main")

    print("Stopping all threads")
    for t in threading.enumerate():
        if t.name != "MainThread":
            print("Thread {} is alive! Is daemon: {}".format(t.name, t.daemon))
            t.join(timeout=1)
            print("Thread joined")

    print("Exiting script")

    # Making sure framework really exits
    os._exit(os.EX_OK)


def evaluate_sequential(args, runner):

    for _ in range(args.test_nepisode):
        runner.run(test_mode=True)

    if args.save_replay:
        runner.save_replay()

    runner.close_env()

def evaluate_sequential_wolfpack(args, runner):

    for _ in range(args.test_nepisode):
        runner.run_wolfpack_attacker(test_mode=True)

    if args.save_replay:
        runner.save_replay()
    runner.close_env()
def evaluate_random(args, runner):

    for _ in range(args.test_nepisode):
        runner.run_randomattack(test_mode=True)

    if args.save_replay:
        runner.save_replay()
    runner.close_env()
def evaluate_sequential_continuous(args, runner):
    """Evaluate continuous attack for Causal_q_learner."""
    for _ in range(args.test_nepisode):
        runner.run_continuous_attack(test_mode=True)

    if args.save_replay:
        runner.save_replay()

    runner.close_env()
    
def run_sequential_rule(args, logger):

    # Init runner so we can get env info
    runner = r_REGISTRY[args.runner](args=args, logger=logger)

    # Set up schemes and groups here
    env_info = runner.get_env_info()
    args.n_agents = env_info["n_agents"]
    args.n_actions = env_info["n_actions"]
    args.state_shape = env_info["state_shape"]
    args.obs_shape = env_info["obs_shape"]

    # Use max of num_followup_agents and num_followup_agents_wall for scheme
    max_followup_agents = max(
        getattr(args, 'num_followup_agents', 1),
        getattr(args, 'num_followup_agents_wall', 1)
    )
       
    scheme = {
        "state": {"vshape": env_info["state_shape"]},
        "obs": {"vshape": env_info["obs_shape"], "group": "agents"},
        "hidden_states": {"vshape": args.rnn_hidden_dim, "group": "agents"},
        "initial_agent": {"vshape": 1, "dtype": th.long}, # selected_agents
        "followup_agents": {"vshape": max_followup_agents, "dtype": th.long}, # selected_agents_next (supports both wolfpack and continuous)
        "actions": {"vshape": (1,), "group": "agents", "dtype": th.long},
        "forced_actions": {"vshape": (1,), "group": "agents", "dtype": th.long},
        "attacker_actions" : {"vshape": (1,), "group": "agents", "dtype": th.long},
        "avail_actions": {"vshape": (env_info["n_actions"],), "group": "agents", "dtype": th.int},
        "reward": {"vshape": (1,)},
        "terminated": {"vshape": (1,), "dtype": th.uint8},
    }
    groups = {
        "agents": args.n_agents
    }
    preprocess = {
        "actions": ("actions_onehot", [OneHot(out_dim=args.n_actions)]),
        "forced_actions": ("forced_actions_onehot", [OneHot(out_dim=args.n_actions)]),
        "attacker_actions": ("attacker_actions_onehot", [OneHot(out_dim=args.n_actions)]),
    }

    buffer = ReplayBuffer(scheme, groups, args.buffer_size, env_info["episode_limit"] + 1,
                          preprocess=preprocess,
                          device="cpu" if args.buffer_cpu_only else args.device)

    # Setup multiagent controller here
    mac = mac_REGISTRY[args.mac](buffer.scheme, groups, args)



    # Learner
    if args.learner == "WALL_q_learner":
        qdifference_transformer = Qdifference_Transformer(args)
        planning_transformer = Planning_Transformer(args)
        runner.setup(scheme=scheme, groups=groups, preprocess=preprocess, mac=mac, qdifference_transformer=qdifference_transformer, planning_transformer=planning_transformer,obs_predictor=None)
        learner = le_REGISTRY[args.learner](mac, buffer.scheme, logger, args, qdifference_transformer, planning_transformer)
    elif args.learner == "q_learner":
        qdifference_transformer = None
        planning_transformer = None
        runner.setup(scheme=scheme, groups=groups, preprocess=preprocess, mac=mac, qdifference_transformer=qdifference_transformer, planning_transformer=planning_transformer,obs_predictor=None)

        learner = le_REGISTRY[args.learner](mac, buffer.scheme, logger, args)
    elif args.learner == "Causal_q_learner":
        qdifference_transformer = Qdifference_Transformer(args)
        planning_transformer = Planning_Transformer(args)
        ObsPredictor = ObsPredictorMLP(args)
        runner.setup(scheme=scheme, groups=groups, preprocess=preprocess, mac=mac, qdifference_transformer=qdifference_transformer, planning_transformer=planning_transformer,obs_predictor=ObsPredictor)
        learner = le_REGISTRY[args.learner](mac, buffer.scheme, logger, args, qdifference_transformer, planning_transformer,ObsPredictor)
    if args.use_cuda:
        learner.cuda()

    if args.pretrain == True:
        model_path = f"pretrain_model/qmix/" + args.env_args["map_name"]
        learner.load_models(model_path)
        print("pretrain")

    runner.setup_mac_for_attack(mac)
    runner.setup_learner(learner)
    
    if args.checkpoint_path != "":
        timesteps = []
        timestep_to_load = 0

        if not os.path.isdir(args.checkpoint_path):
            logger.console_logger.info("Checkpoint directiory {} doesn't exist".format(args.checkpoint_path))
            return

        # Go through all files in args.checkpoint_path
        for name in os.listdir(args.checkpoint_path):
            full_name = os.path.join(args.checkpoint_path, name)
            # Check if they are dirs the names of which are numbers
            if os.path.isdir(full_name) and name.isdigit():
                timesteps.append(int(name))

        if args.load_step == 0:
            # choose the max timestep
            timestep_to_load = max(timesteps)
        else:
            # choose the timestep closest to load_step
            timestep_to_load = min(timesteps, key=lambda x: abs(x - args.load_step))

        model_path = os.path.join(args.checkpoint_path, str(timestep_to_load))

        logger.console_logger.info("Loading model from {}".format(model_path))
        
        learner.load_models(model_path)
        learner.load_attackers(model_path)
        
        runner.t_env = timestep_to_load

        if args.evaluate or args.save_replay:
            runner.log_train_stats_t = runner.t_env
            if args.learner == "q_learner":
                evaluate_sequential(args, runner)
            elif args.learner == "WALL_q_learner":
                evaluate_sequential(args, runner)
                evaluate_sequential_wolfpack(args, runner)
                evaluate_sequential_continuous(args, runner)
            elif args.learner == "Causal_q_learner":
                evaluate_sequential(args, runner)               
                evaluate_sequential_wolfpack(args, runner)
                evaluate_sequential_continuous(args, runner)
                evaluate_random(args, runner)
            # Print the aggregated evaluation stats before exiting
            logger.log_stat("episode", runner.t_env, runner.t_env)
            logger.print_recent_stats()
            logger.console_logger.info("Finished Evaluation")

            return

    # start training
    episode = 0
    last_test_T = -args.test_interval - 1
    last_log_T = 0
    model_save_time = 0

    start_time = time.time()
    last_time = start_time

    logger.console_logger.info("Beginning training for {} timesteps".format(args.t_max))

    while runner.t_env <= args.t_max:

        # Run for a whole episode at a time
        if args.learner == "q_learner":
            episode_batch = runner.run(test_mode=False)
        elif args.learner == "WALL_q_learner":
            episode_batch = runner.run_wolfpack_attacker(test_mode=False)
        elif args.learner == "Causal_q_learner":
            episode_batch = runner.run_continuous_attack(test_mode=False)

        buffer.insert_episode_batch(episode_batch)

        if buffer.can_sample(args.batch_size):
            episode_sample = buffer.sample(args.batch_size)

            # Truncate batch to only filled timesteps
            max_ep_t = episode_sample.max_t_filled()
            episode_sample = episode_sample[:, :max_ep_t]

            if episode_sample.device != args.device:
                episode_sample.to(args.device)

            learner.train(episode_sample, runner.t_env, episode)

        # Execute test runs once in a while
        n_test_runs = max(1, args.test_nepisode // runner.batch_size)
        if (runner.t_env - last_test_T) / args.test_interval >= 1.0:

            logger.console_logger.info("t_env: {} / {}".format(runner.t_env, args.t_max))
            logger.console_logger.info("Estimated time left: {}. Time passed: {}".format(
                time_left(last_time, last_test_T, runner.t_env, args.t_max), time_str(time.time() - start_time)))
            last_time = time.time()

            last_test_T = runner.t_env
            for _ in range(n_test_runs):
                if args.learner == "WALL_q_learner":
                    runner.run(test_mode=True)
                    runner.run_wolfpack_attacker(test_mode=True)
                elif args.learner == "q_learner":
                    runner.run(test_mode=True)
                elif args.learner == "Causal_q_learner":
                    runner.run(test_mode=True)
                    runner.run_wolfpack_attacker(test_mode=True)
                    runner.run_continuous_attack(test_mode=True)

        if args.save_model and (runner.t_env - model_save_time >= args.save_model_interval or model_save_time == 0):
            model_save_time = runner.t_env
            save_path = os.path.join(args.local_results_path, "models", args.unique_token, str(runner.t_env))
            os.makedirs(save_path, exist_ok=True)
            logger.console_logger.info("Saving models to {}".format(save_path))

            # learner should handle saving/loading -- delegate actor save/load to mac,
            # use appropriate filenames to do critics, optimizer states
            learner.save_models(save_path)

        episode += args.batch_size_run

        if (runner.t_env - last_log_T) >= args.log_interval:
            logger.log_stat("episode", episode, runner.t_env)
            if args.use_wandb:
                wandb.log({"episode": episode}, step=runner.t_env)
            logger.print_recent_stats()
            last_log_T = runner.t_env

    runner.close_env()
    logger.console_logger.info("Finished Training")


def run_sequential_RL(args, logger):
    # Init runner so we can get env info
    runner = r_REGISTRY[args.runner](args=args, logger=logger)

    # Set up schemes and groups here
    env_info = runner.get_env_info()
    args.n_agents = env_info["n_agents"]
    args.n_actions = env_info["n_actions"]
    args.state_shape = env_info["state_shape"]
    args.episode_limit = env_info["episode_limit"]

    # Default/Base scheme
    scheme = {
        "state": {"vshape": env_info["state_shape"]},
        "obs": {"vshape": env_info["obs_shape"], "group": "agents"},
        "actions": {"vshape": (1,), "group": "agents", "dtype": th.long},
        "forced_actions": {"vshape": (1,), "group": "agents", "dtype": th.long},
        "avail_actions": {"vshape": (env_info["n_actions"],), "group": "agents", "dtype": th.int},
        "reward": {"vshape": (1,)},
        "terminated": {"vshape": (1,), "dtype": th.uint8},
    }
    groups = {
        "agents": args.n_agents
    }
    preprocess = {
        "actions": ("actions_onehot", [OneHot(out_dim=args.n_actions)]),
        "forced_actions": ("forced_actions_onehot", [OneHot(out_dim=args.n_actions)]),
    }

    buffer = ReplayBuffer(scheme, groups, args.buffer_size, env_info["episode_limit"] + 1,
                          preprocess=preprocess,
                          device="cpu" if args.buffer_cpu_only else args.device)

    # Setup multiagent controller here
    mac = mac_REGISTRY[args.mac](buffer.scheme, groups, args)

    # Learner
    learner = le_REGISTRY[args.learner](mac, buffer.scheme, logger, args)

    if args.use_cuda:
        learner.cuda()
    if args.pretrain == True:
        model_path = f"pretrain_model/qmix/" + args.env_args["map_name"]
        learner.load_models(model_path)
        print("pretrain")
    if args.checkpoint_path != "":
        timesteps = []
        timestep_to_load = 0

        if not os.path.isdir(args.checkpoint_path):
            logger.console_logger.info("Checkpoint directiory {} doesn't exist".format(args.checkpoint_path))
            return

        # Go through all files in args.checkpoint_path
        for name in os.listdir(args.checkpoint_path):
            full_name = os.path.join(args.checkpoint_path, name)
            # Check if they are dirs the names of which are numbers
            if os.path.isdir(full_name) and name.isdigit():
                timesteps.append(int(name))

        if args.load_step == 0:
            # choose the max timestep
            timestep_to_load = max(timesteps)
        else:
            # choose the timestep closest to load_step
            timestep_to_load = min(timesteps, key=lambda x: abs(x - args.load_step))

        model_path = os.path.join(args.checkpoint_path, str(timestep_to_load))

        logger.console_logger.info("Loading model from {}".format(model_path))
        
        learner.load_models(model_path)

        # set pre-trained model for comparison
        ori_mac = None
        """ori_mac = mac_REGISTRY[args.mac](buffer.scheme, groups, args)
        ori_learner = le_REGISTRY[args.learner](ori_mac, buffer.scheme, logger, args)
        model_path = args.checkpoint_path + args.env_args["map_name"]
        logger.console_logger.info("Loading original model from {}".format(model_path))
        ori_learner.load_models(model_path)
        if args.use_cuda:
            ori_learner.cuda()"""

    # Attacker
    attacker_scheme = {
        "state": {"vshape": args.state_shape},
        "action": {"vshape": (1,), "dtype": th.long},
        "reward": {"vshape": (1,)},
        "shaping_reward": {"vshape": (1,)},
        "terminated": {"vshape": (1,), "dtype": th.uint8},  # terminate if attack num is used or game finish
        "left_attack": {"vshape": (1,)},  # ratio of left attack times
    }
    attacker_groups = None
    attacker_preprocess = {
        "action": ("action_onehot", [OneHot(out_dim=args.n_agents + 1)])
    }

    # Set archive
    archive = Archive(args)
    if args.archive_load_path != "":
        logger.console_logger.info(f"log attacker archive from {args.archive_load_path}")
        archive.load_models(args.archive_load_path)

    test_archive = None
    if args.test_attacker_archive_path != "":
        test_archive = Archive(args)
        logger.console_logger.info(f"log testing attacker archive from {args.test_attacker_archive_path}")
        test_archive.load_models(args.test_attacker_archive_path)
        test_returns, test_won_rates = [], []
        save_test_path = os.path.join(args.local_results_path, "test_results",
                                 args.env_args["map_name"] + f"_{args.attack_num}",
                                 args.unique_token)
        os.makedirs(save_test_path, exist_ok=True)
        save_test_return_path = os.path.join(save_test_path, "test_return")
        save_test_wons_path = os.path.join(save_test_path, "test_won")


    population = Population(args)
    population.setup_buffer(attacker_scheme, attacker_groups, attacker_preprocess)

    runner.setup(scheme, groups, preprocess, attacker_scheme, attacker_groups, attacker_preprocess)


    logger.console_logger.info(f"start with device {args.device}")

    if test_archive is not None:
        logger.console_logger.info(f"save testing results")
        r, w = test_archive.long_eval(mac, runner, logger, 1, 5)
        test_returns.append(r)
        test_won_rates.append(w)
        logger.console_logger.info(f"save info in {save_test_path}")
        np.savetxt(save_test_return_path, test_returns)
        np.savetxt(save_test_wons_path, test_won_rates)

    if args.start_eval:
        logger.console_logger.info(f"start eval")

        logger.console_logger.info(f"robust trained agents")
        save_path = os.path.join(args.local_results_path, "eval_results",
                                 args.env_args["map_name"] + f"_{args.attack_num}",
                                 args.unique_token, "start_eval_robust")
        run_evaluate(args, test_archive, mac, runner, logger, save_path)

        if ori_mac is not None:
            logger.console_logger.info(f"evaluating original ego agents for comparison")
            save_path = os.path.join(args.local_results_path, "eval_results",
                                     args.env_args["map_name"] + f"_{args.attack_num}",
                                     args.unique_token, "start_eval_original")
            run_evaluate(args, test_archive, ori_mac, runner, logger, save_path)
        return None

    for gen in range(args.generation):
        print(f"Start generation {gen + 1}/{args.generation} attacker and ego-agents training")

        if gen >= args.finetune_gen: # do not train attackers
            args.fine_tune = True

        selected_attackers = archive.select(gen)
        population.reset(selected_attackers)
        if args.use_cuda:
            population.cuda()

        if gen == 0:
            runner.setup_mac(mac)
            wa_returns, wa_wons = [], []
            for _ in range(args.default_nepisode):
                r, w, _ = runner.run_without_attack()
                #r, w, _ = runner.run_random_attack(True)
                wa_returns.append(r)
                wa_wons.append(w)
            print(f"default return mean: {np.mean(wa_returns)}, default battle won mean: {np.mean(wa_wons)}")

        for train_step in range(args.population_train_steps):
            if gen == 0 and train_step == 0:
                for attacker_id, attacker in enumerate(population.attackers):
                    mac.set_attacker(attacker)
                    runner.setup_mac(mac)
                    for episode_idx in range(args.attack_batch_size // args.pop_size + 1):
                        gen_mask = ((episode_idx % 2) != 0) and not args.fine_tune
                        ego_epi_batch, attacker_epi_batch, mixed_points, attack_cnt, _, _ = runner.run(
                            test_mode=False, gen_mask=gen_mask)
                        if gen_mask == False:
                            buffer.insert_episode_batch(ego_epi_batch)
                        population.store(attacker_epi_batch, mixed_points, attack_cnt, attacker_id)

            #print(f"collect data at generation: {gen + 1}/{args.generation}; "
            #      f"train_step: {train_step + 1}/{args.population_train_steps}")

            for attacker_id, attacker in enumerate(population.attackers):
                mac.set_attacker(attacker)
                runner.setup_mac(mac)
                gen_mask = ((episode_idx % 2) != 0) and not args.fine_tune
                ego_epi_batch, attacker_epi_batch, mixed_points, attack_cnt, epi_return, _ = runner.run(test_mode=False,
                                                                                                        gen_mask=gen_mask)
                if gen_mask == False:
                    buffer.insert_episode_batch(ego_epi_batch)
                population.store(attacker_epi_batch, mixed_points, attack_cnt, attacker_id)

            train_ok = True
            if not args.fine_tune and train_step < args.population_train_steps//2:
                for _ in range(args.population_train_num):
                    train_ok = population.train(gen, train_step)
                    if train_ok == False:
                        break
            if train_ok == False:
                break

            if buffer.can_sample(args.batch_size) and train_step >= args.population_train_steps//2:
                logger.console_logger.info("Training ego agents")
                train_num = args.pop_size * 2 if args.ego_train_step == None else args.ego_train_step
                for _ in range(train_num):
                    episode_sample = buffer.sample(args.batch_size)

                    # Truncate batch to only filled timesteps
                    max_ep_t = episode_sample.max_t_filled()
                    episode_sample = episode_sample[:, :max_ep_t]

                    if episode_sample.device != args.device:
                        episode_sample.to(args.device)
                    learner.train(episode_sample, gen, train_step)
                learner._update_targets()

        #assert test_archive is not None
        if (gen+1) % 4 == 0 and test_archive is not None:
            #logger.console_logger.info(f"save testing results in {save_test_path}")
            r, w = test_archive.long_eval(mac, runner, logger, 1, 5)
            test_returns.append(r)
            test_won_rates.append(w)
            np.savetxt(save_test_return_path, test_returns)
            np.savetxt(save_test_wons_path, test_won_rates)

        if train_ok == False:
            continue

        last_attack_points, last_mean_return, last_mean_won = population.get_behavior_info(mac, runner)

        # update archive behaviors since ego-agents change
        if not args.fine_tune:
            archive.update_behavior(mac, runner)

            archive.update(population, last_attack_points, last_mean_return, last_mean_won)

        if (gen + 1) % args.save_archive_interval == 0:
            # save attackers
            save_path = os.path.join(args.local_results_path, "robust_attacker_archive",
                                     args.env_args["map_name"] + f"_{args.attack_num}", args.unique_token, str(gen + 1))
            print(f"save generations {gen + 1} in {save_path}")
            os.makedirs(save_path, exist_ok=True)
            logger.console_logger.info("Saving models to {}".format(save_path))
            archive.save_models(save_path)

            # save ego-agents
            save_path = os.path.join(args.local_results_path, "ego_agents",
                                     args.env_args["map_name"] + f"_{args.attack_num}",
                                     args.unique_token, str(gen + 1))
            os.makedirs(save_path, exist_ok=True)
            logger.console_logger.info("Saving ego-agents models to {}".format(save_path))
            learner.save_models(save_path)

        if (gen + 1) % args.long_eval_interval == 0:
            archive.long_eval(mac, runner, logger)

        if (gen + 1) % args.attack_nepisode:
            logger.print_recent_stats()

        if (gen + 1) % 10 == 0:
            wa_returns, wa_wons = [], []
            for _ in range(args.default_nepisode):
                x, y, _ = runner.run_without_attack()
                wa_returns.append(x)
                wa_wons.append(y)
            #print(f"without attack, recent returns {np.mean(wa_returns)}, recent battle won {np.mean(wa_wons)}")
            logger.print_recent_stats()

    if test_archive is not None:
        save_path = os.path.join(args.local_results_path, "eval_results",
                                 args.env_args["map_name"] + f"_{args.attack_num}",
                                 args.unique_token, "end_eval_attack")
        run_evaluate(args, test_archive, mac, runner, logger, save_path)

    runner.close_env()
    logger.console_logger.info("Finished Training")


def run_evaluate(args, archive, mac, runner, logger, save_path=None):
    archive.long_eval(mac, runner, logger, 1, args.eval_num, save_path=save_path)
def args_sanity_check(config, _log):

    # set CUDA flags
    # config["use_cuda"] = True # Use cuda whenever possible!
    if config["use_cuda"] and not th.cuda.is_available():
        config["use_cuda"] = False
        _log.warning("CUDA flag use_cuda was switched OFF automatically because no CUDA devices are available!")

    if config["test_nepisode"] < config["batch_size_run"]:
        config["test_nepisode"] = config["batch_size_run"]
    else:
        config["test_nepisode"] = (config["test_nepisode"]//config["batch_size_run"]) * config["batch_size_run"]

    return config
