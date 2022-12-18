import shutil
from pathlib import Path
from main import CQLAgent


def test(env_id, load_dir):
    monitor_dir = Path(__file__).parent / 'mp4'
    if monitor_dir.exists():
        shutil.rmtree(monitor_dir)

    agent = CQLAgent(env_id=env_id)
    agent.load(load_dir=load_dir)

    for i in range(3):  # default=10
        print(agent.rollout())

    print('Finished')


def main():
    env_id = "Breakout"
    test(env_id=env_id, load_dir='checkpoints/')


if __name__ == '__main__':
    main()
