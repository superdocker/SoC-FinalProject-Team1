# Title

- HLS를 이용한 Sparse Convolution용 CSR기반 Rule generator 설계
- 한양대학교 2021-2
- SoC설계방법론 (최정욱 교수님)
- Team 1: 이장환, 박성민, 이민재 

# Setup & Run

## Versions  
- No dataflow optimization ```csr/csr_onboard_init/```  
- With dataflow optimization ```csr/csr_dataflow_rowloop/```  

|           | No dataflow optimization | With dataflow optimization |
|-----------|--------------------------|----------------------------|
| SW_EMU    | Available                | Available                  |
| HW_EMU    | Available                | Available                  |
| Run on F1 | Available                | Not Available              |

## Commands  
```bash
# Setup with codes in csr/csr_onboard_init/
# Another codes, csr/csr_onboard_init/ is available

# In AWS instance
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
# Run on F1 instance
make TARGET=hw DEVICE=$AWS_PLATFORM all
aws configure
aws s3 ls {YOUR_BUCKET_NAME}
$VITIS_DIR/tools/create_vitis_afi.sh -xclbin=./build_dir.hw.xilinx_aws-vu9p-f1_shell-v04261818_201920_2/vadd.xclbin -s3_bucket={YOUR_BUCKET_NAME} -s3_dcp_key=dcp -s3_logs_key=logs
# Check AFI ID
cat *_afi_id.txt
aws ec2 describe-fpga-images --fpga-image-ids afi-[AFI ID]
# when state changes pending to available, move hello_world and vadd.awsxclbin to F1 instance

# On F1 board
cd $AWS_FPGA_REPO_DIR
source vitis_setup.sh
source vitis_runtime_setup.sh
cd YOUR_DIR
# Run
chmod +x hello_world
./hello_world vadd.awsxclbin 
```

# About codes

Explanation of codes is in each repo directory.  
- No dataflow optimization: https://github.com/superdocker/SoC-FinalProject-Team1/tree/main/csr#no-dataflow-optimization
- With dataflow optimization: https://github.com/superdocker/SoC-FinalProject-Team1/tree/main/csr#with-dataflow-optimization
