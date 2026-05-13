
import os
import argparse
from loguru import logger
from network_helper import NetworkHelper
from agent_helper import AgentHelper
from structures import NetworkHelperParameters, AgentHelperParameters


if __name__ == "__main__":

    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Run the CoPTER experiment.")
    parser.add_argument("-p", "--ns3_socket", type=int, default=5555, help="ns3_socket: The port number for the ns3-gym environment.")
    parser.add_argument("-e", "--exp_name", type=str, default="copter_experiment", help="exp_name: The name of the experiment for logging.")
    parser.add_argument("-r", "--override_name", type=str, default=None, help="override_name: Override the experiment name for loading models.")
    parser.add_argument("-m", "--mode", type=str, choices=["ACC", "CoPTER"], default="ACC", help="mode: The mode of the agent, either 'ACC' or 'CoPTER'.")
    parser.add_argument("-f", "--fmap_dir", type=str, default=None, help="fmap_dir: The directory for the feature map, required if mode is 'CoPTER'.")
    parser.add_argument("-o", "--online", action="store_true", help="online: Whether to run the agent in online mode.")
    parser.add_argument("-d", "--model_dir", type=str, default="models", help="model_dir: The directory where the model is stored.")
    parser.add_argument("-s", "--static_steps", type=int, default=4, help="static_steps: The number of initial steps to keep the actions static.")
    parser.add_argument("-i", "--train_intervals", type=int, default=8, help="train_intervals: The number of intervals for training the agent.")
    parser.add_argument("-b", "--switch_buffer", type=int, default=10000, help="switch_buffer: The size of switch buffer in bytes.")
    args = parser.parse_args()

    # Print the parsed arguments
    logger.info(f"Parsed arguments: {args}")

    # Initialize Logger
    logger.add(args.exp_name + "_log/copter_{time}.log", level="INFO", rotation="5 MB")
    logger.info(f"Copter Starting Running at {os.getcwd()}")

    # Initialize NetworkHelper
    network_helper_params = NetworkHelperParameters(port_states=6, port_actions=3, state_observations=3, switch_buffer_size=args.switch_buffer)
    
    network_helper = NetworkHelper(ns3_socket=args.ns3_socket, nhp=network_helper_params)
    
    agent_helper_params = AgentHelperParameters()

    agent_helper = AgentHelper(node_number=network_helper.get_n_port(), ahp=agent_helper_params, model_dir=args.model_dir, mode=args.mode, 
                               exp_name=args.exp_name, online=args.online, network_helper=network_helper, fmap_dir=args.fmap_dir)
    
    agent_helper.load(args.override_name)

    current_step = 0

    try:
        while True:
            if current_step == 0:
                # Get the initial observation for all ports
                network_helper.monitor(current_step)

            elif current_step < max(4, args.static_steps):
                # For the first few steps, we keep their actions as what the environment provides.
                for port_idx in range(network_helper.get_n_port()):
                    paras = network_helper.get_port_current_parameters(port_idx)
                    network_helper.configurator(current_step, port_idx, paras)
                network_helper.monitor(current_step)

            else:
                # Get the current states for all ports. NOTE: `Observation` is different from `State` in CoPTER.
                port_states = [
                    network_helper.get_port_current_state_list(port_idx)
                    for port_idx in range(network_helper.get_n_port())
                ]

                # Decide the actions for all ports with AgenHelper
                paras, actions = agent_helper.decide(port_states)

                # Cofigure the actions for each port
                for port_idx, parameter in enumerate(paras):
                    network_helper.configurator(current_step, port_idx, parameter)

                # Enforce the updated parameters
                network_helper.monitor(current_step)
            # 步骤1：先执行monitor，确保获取端口标识
            # done = network_helper.monitor(current_step)

            # # 步骤2：仅在标识获取完成后（current_step≥1）执行决策
            # if current_step >= 1 and network_helper.port_identifier_map:
            #     # 获取端口状态
            #     port_states = [
            #         network_helper.get_port_current_state_list(port_idx)
            #         for port_idx in range(network_helper.get_n_port())
            #     ]
            #     # 调用decide（此时会触发fmap延迟加载）
            #     paras, actions = agent_helper.decide(port_states, epsi=args.epsilon)
                
            #     # 配置动作
            #     for port_idx, parameter in enumerate(paras):
            #         network_helper.configurator(current_step, port_idx, parameter)

                # 只有在线模式才记录经验（如果offline模式不需要记录，可添加此判断）
                if args.online:

                    # Record the states and actions in the agent's replay buffer
                    for port_idx in range(network_helper.get_n_port()):
                        # Get the current (new) states, (last) states, actions (in dice format), and rewards
                        current_state = network_helper.get_port_current_state_list(port_idx)
                        last_state = network_helper.get_port_last_state_list(port_idx)
                        action = actions[port_idx]
                        # reward = network_helper.get_port_current_reward(port_idx)
                        # Add the experience to the agent's replay buffer
                        # agent_helper.record(port_idx, last_state, action, reward, current_state)
                        agent_helper.record(port_idx, last_state, action, current_state)

                # Train the agent every `train_intervals` steps
                if args.online and current_step % args.train_intervals == 0:
                    agent_helper.sync()
                    agent_helper.train(current_step)
                    # Save the agents' model and replay buffers whenever training is done.
                    agent_helper.save()

            current_step += 1

    except KeyboardInterrupt:
        print("Ctrl-C -> Exit")
        
    finally:
        network_helper.close_env()
        print("Done")