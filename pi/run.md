
##### Build Docker
docker build -t ml-sha/nudge_docker .

##### Run Docker
docker run --gpus all -it -v /home/ml-jsha/storage:/NUDGE/storage --rm ml-sha/nudge_docker

#### Train Atari 2600 Games

``` 
python -m pi.play --teacher_agent pretrained -m Asterix --hardness 0 --with_explain --device 9
```