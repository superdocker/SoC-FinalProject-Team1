# Title

- 한양대학교 2021-2
- SoC설계방법론 (최정욱 교수님)
- Team 1  
- Authors: Janghwan Lee, Seongmin Park, Minjae Lee

```bash
cd $AWS_FPGA_REPO_DIR
source vitis_setup.sh
cd $VITIS_DIR/examples/xilinx
git clone https://github.com/superdocker/SoC-FinalProject-Team1.git
cp -r SoC-FinalProject-Team1/csr/csr_onboard_init .
cd csr_onboard_init/
# software emulation
make run TARGET=sw_emu DEVICE=$AWS_PLATFORM all
# hardware emulation
make run TARGET=hw_emu DEVICE=$AWS_PLATFORM all

make TARGET=hw DEVICE=$AWS_PLATFORM all
aws configure
aws s3 ls hls-student01-bucket
$VITIS_DIR/tools/create_vitis_afi.sh -xclbin=./build_dir.hw.xilinx_aws-vu9p-f1_shell-v04261818_201920_2/vadd.xclbin -s3_bucket=hls-student01-bucket -s3_dcp_key=dcp -s3_logs_key=logs
cat *_afi_id.txt
aws ec2 describe-fpga-images --fpga-image-ids afi-[AFI ID]
when state changes pending to available, move hello_world and vadd.awsxclbin to F1 instance

# On F1 board
cd $AWS_FPGA_REPO_DIR
source vitis_setup.sh
source vitis_runtime_setup.sh
cd YOUR_DIR

chmod +x hello_world
./hello_world vadd.awsxclbin 
```
