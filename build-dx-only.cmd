rd /s build

call "C:\Program Files (x86)\Microsoft Visual Studio\2017\Community\VC\Auxiliary\Build\vcvarsall.bat" amd64
meson.exe build --buildtype release -Ddefault_library=static -Deigen=true

pause


cd build

ninja

pause