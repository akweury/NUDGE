
##### Build Docker
docker build -t ml-sha/nudge_docker .

##### Run Docker
docker run --gpus all -it -v /home/ml-jsha/storage:/NUDGE/storage --rm ml-sha/nudge_docker

#### Train Atari 2600 Games

``` 
python -m atari_py.import_roms /NUDGE/storage/ROMS

```

``` 
python -m pi.play -m Asterix --with_explain --teacher_game_nums 1000 --device 4
python -m pi.play --teacher_agent pretrained -m Boxing --with_explain --device 10 --teacher_game_nums 100
```