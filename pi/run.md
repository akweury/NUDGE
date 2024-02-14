
##### Build Docker
docker build -t ml-sha/nudge_docker .

##### Run Docker
docker run --gpus all -it -v /home/ml-jsha/storage:/NUDGE/storage --rm ml-sha/nudge_docker

#### Train Atari 2600 Games

``` 
python -m atari_py.import_roms /storage/ROMS

```

``` 
python -m pi.play --teacher_agent pretrained -m asterix --hardness 0 --with_explain --device 9
```