## Docker build and run

There are currently 2 types of containers:
- rocm (uses amd's rocm tools to use cuda acceleration)
- cpu (no cuda)


### Graphical output from docker
If required to see gui output from the container:
`xhost +local:docker`

### rocm
To build and run the docker:

`sudo docker compose up rocm`

Attach to it from another window:
`sudo docker exec -it rocm bash`

Or run 

`sudo docker compose run rocm bash`

to run and attach to the window in one terminal.

### cpu
```
docker compose up cpu
docker exec -it cpu bash
```

or

```
docker compose run cpu bash
```

## Training

Train a rf-detr nano segmentation model on the dataset specified after `-d`.
Run this from the `segmentation` directory.
```
python train.py -d ../data/dataset/312x312-no-bg-no-artificial/ -s nano
```

## Testing

Test the model by using the `eval.py` script, e.g.
```
python segmentation/eval.py -m runs/segm/output2/checkpoint_best_total.pth -i data/images/2025-10-07/image352.png -s nano
```