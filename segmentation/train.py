# from rfdetr import RFDETRSegLarge
from pathlib import Path
import argparse
import os

def main(args):
    data_path = Path(args.dataset_path).resolve()

    # Import model based on the model size required
    if args.model_size == "nano":
        from rfdetr import RFDETRSegNano
        model = RFDETRSegNano()
    elif args.model_size == "small":
        from rfdetr import RFDETRSegSmall
        model = RFDETRSegSmall()
    elif args.model_size == "medium":
        from rfdetr import RFDETRSegMedium
        model = RFDETRSegMedium()
    elif args.model_size == "large":
        from rfdetr import RFDETRSegLarge
        model = RFDETRSegLarge()
    else:
        raise ValueError("Invalid model size. Options: nano, small, medium, large")

    print(f"****************Model: {args.model_size}****************")
    print(f"****************Dataset: {data_path}****************")

    # Output directory
    if args.output_dir is not None:
        output_path = str(Path(args.output_dir).resolve())
        os.makedirs(output_path, exist_ok=True)
    else:
        output_dir = Path(__file__).parent.parent / "runs" / "segm"
        if not output_dir.exists():
            os.makedirs(output_dir)
            output_n = 1
        else:
            output_n = max([int(f.name.split("output")[1]) for f in output_dir.iterdir() if f.name.startswith("output")])
    
        output_path = str((output_dir / f"output{output_n+1}").resolve())
        print(f"****************Output directory: {output_path}****************")
        os.makedirs(output_path, exist_ok=True)
        # Look for output folder in the output_dir and find the latest iteration number

    # Resume training
    if args.resume:
        if args.resume_checkpoint_path is None:
            raise ValueError("resume_checkpoint_path is required when resuming training")
        else:
            checkpoint_path = args.resume_checkpoint_path
            print(f"****************Resuming training from checkpoint: {checkpoint_path}****************")
    else:
        checkpoint_path = None

    history = []

    # Save comment
    if args.comment is not None:
        with open(output_path + "/comment.txt", "w") as f:
            f.write(str(args.comment))

    def callback2(data):
        history.append(data)

    if "on_fit_epoch_end" in model.callbacks:
        model.callbacks["on_fit_epoch_end"].append(callback2)
    elif "on_epoch_end" in model.callbacks:
        model.callbacks["on_epoch_end"].append(callback2)

    print(f"Training params: {args}")

    model.train(
    dataset_dir=str(data_path),
    epochs=args.epochs,
    batch_size=args.batch_size,
    grad_accum_steps=args.grad_accum_steps,
    lr=args.lr,
    output_dir=str(output_path),
    early_stopping=args.early_stopping,
    tensorboard=args.tensorboard,
    resume=str(args.resume_checkpoint_path) if args.resume and args.resume_checkpoint_path.exists() else None,
    )


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--dataset_path', type=str, default="./dataset", help='Dataset directory. Default: ./dataset', required=True)
    parser.add_argument('-s', '--model_size', type=str, default="small", help='Model size. Options: nano, small, medium, large. Default: small', required=False)
    parser.add_argument('-ep', '--epochs', type=int, default=40, help='Number of epochs. Default: 40', required=False)
    parser.add_argument('-b', '--batch_size', type=int, default=3, help='Batch size. Default: 3', required=False)
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate. Default: 1e-4', required=False)
    parser.add_argument('--grad_accum_steps', type=int, default=4, help='Gradient accumulation steps. Default: 4', required=False)
    parser.add_argument('-o', '--output_dir', type=str, default=None, help='Output directory for results. Default: None', required=False)
    parser.add_argument('--early_stopping', type=bool, default=True, help='Early stopping. Default: True', required=False)
    parser.add_argument('--tensorboard', type=bool, default=True, help='Tensorboard. Default: True', required=False)
    parser.add_argument('--resume', type=bool, default=False, help='Resume from checkpoint. Default: False', required=False)
    parser.add_argument('-rcp', '--resume_checkpoint_path', type=str, default=None, help='Path to the model checkpoint. Default: None', required=False)
    parser.add_argument('-c', '--comment', type=str, default=None, help='Comment. Default: None', required=False)

    return parser.parse_args()


if __name__ == "__main__":
     args = parse_args()
     main(args)
    


# df = pd.DataFrame(history)
# print(f"df columns: {df.columns}")

# plt.figure(figsize=(12, 8))

# plt.plot(
# 	df['epoch'],
# 	df['train_loss'],
# 	label='Training Loss',
# 	marker='o',
# 	linestyle='-'
# )

# plt.plot(
# 	df['epoch'],
# 	df['test_loss'],
# 	label='Validation Loss',
# 	marker='o',
# 	linestyle='--'
# )

# plt.title('Train/Validation Loss over Epochs')
# plt.xlabel('Epoch')
# plt.ylabel('Loss')
# plt.legend()
# plt.grid(True)

# plt.show()