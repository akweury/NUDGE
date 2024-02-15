
##### Build Docker
docker build -t ml-sha/nudge_docker .

##### Run Docker
docker run --gpus all -it -v /home/ml-jsha/storage:/NUDGE/storage --rm ml-sha/nudge_docker

#### Train Atari 2600 Games

``` 
python -m atari_py.import_roms /NUDGE/storage/ROMS

```

``` 
python -m pi.play --teacher_agent pretrained -m Asterix --hardness 0 --with_explain --teacher_game_nums 30 --device 9
python -m pi.play --teacher_agent pretrained -m Boxing --hardness 0 --with_explain --teacher_game_nums 30 --device 9 --teacher_game_nums 100
```