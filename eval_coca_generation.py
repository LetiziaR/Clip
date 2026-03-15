from evaluation.args import parse_args
from evaluation.run import run_evaluation


def main():
    args = parse_args()
    run_evaluation(args)


if __name__ == "__main__":
    main()
