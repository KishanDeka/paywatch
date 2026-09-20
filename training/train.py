from paywatch.train import parser, train

if __name__ == "__main__":
    train(parser().parse_args())
