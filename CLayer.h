#pragma once
#include "Imagelib.h"
#include "CTensor.h"

#define MEAN_INIT 0
#define LOAD_INIT 1

// Layer는 tensor를 입/출력으로 가지며, 특정 operation을 수행하는 Convolutional Neural Netowork의 기본 연산 단위


class Layer {
protected:
	int fK; // kernel size in K*K kernel
	int fC_in; // number of channels
	int fC_out; //number of filters
	string name;
public:
	Layer(string _name, int _fK, int _fC_in, int _fC_out) : name(_name), fK(_fK), fC_in(_fC_in), fC_out(_fC_out) {}
	virtual ~Layer() {}; //가상소멸자 (참고: https://wonjayk.tistory.com/243)
	virtual Tensor3D* forward(const Tensor3D* input) = 0;
	//	virtual bool backward() = 0;
	virtual void print() const = 0;
	virtual void get_info(string& _name, int& _fK, int& _fC_in, int& _fC_out) const = 0;
};


class Layer_ReLU : public Layer {
public:
	Layer_ReLU(string _name, int _fK, int _fC_in, int _fC_out)
		: Layer(_name, _fK, _fC_in, _fC_out) 
	{
		// (구현할 것)//////////////////////////////////////////////////////////////////////
		// 동작1: Base class의 생성자를 호출하여 맴버 변수를 초기화 할 것(반드시 initialization list를 사용할 것)
	}
	~Layer_ReLU() {}
	Tensor3D* forward(const Tensor3D* input) override {
		// (구현할 것)
		// 동작1: input tensor에 대해 각 element x가 양수이면 그대로 전달, 음수이면 0으로 output tensor에 전달할것    
		// 동작2: 이때, output tensor는 동적할당하여 주소값을 반환할 것
		// 함수1: Tensor3D의 맴버함수인 get_info(), get_elem(), set_elem()을 적절히 활용할 것
		int H, W, C;
		input->get_info(H, W, C);

		Tensor3D* output = new Tensor3D(H, W, C);
		for (int h = 0; h < H; h++) {
			for (int w = 0; w < W; w++) {
				for (int c = 0; c < C; c++) {
					double val = input->get_elem(h, w, c);
					output->set_elem(h, w, c, (val > 0) ? val : 0.0);
				}
			}
		}
		cout << name << " is finished" << endl;
		return output;
	};
	void get_info(string& _name, int& _fK, int& _fC_in, int& _fC_out) const override {
		// (구현할 것)//////////////////////////////////////////////////////////////////////
		// 동작: Tensor3D의 get_info()와 마찬가지로 맴버 변수들을 pass by reference로 외부에 전달
		_name = name;
		_fK = fK;
		_fC_in = fC_in;
		_fC_out = fC_out;
	}
	void print() const override {
		// (구현할 것)//////////////////////////////////////////////////////////////////////
		// 동작: Tensor3D의 print()와 마찬가지로 차원의 크기를 화면에 출력
		cout << "Layer: " << name << " (ReLU) "
			<< "Kernel=" << fK
			<< " Cin=" << fC_in
			<< " Cout=" << fC_out << endl;
	}
};



class Layer_Conv : public Layer {
private:
	string filename_weight;
	string filename_bias;
	double**** weight_tensor; // fK x fK x _fC_in x _fC_out 크기를 가지는 4차원 배열
	double*  bias_tensor;     // _fC_out 크기를 가지는 1차원 배열 (bias는 각 filter당 1개 존재) 
public:
	Layer_Conv(string _name, int _fK, int _fC_in, int _fC_out, int init_type, string _filename_weight = "", string _filename_bias = "")
	:Layer(_name, _fK, _fC_in, _fC_out), 
          filename_weight(_filename_weight), filename_bias(_filename_bias)
	{
		// (구현할 것)//////////////////////////////////////////////////////////////////////
		// 동작1: initialization list와 base class의 생성자를 이용하여 맴버 변수를 초기화 할 것
		// 동작2: filename_weight와 filename_bias는 LOAD_INIT 모드일 경우 해당 파일로부터 가중치/바이어스를 불러옴
		// 동작3: init() 함수는 init_type를 입력으로 받아 가중치를 초기화 함 
		// 함수1: dmatrix4D()와 dmatrix1D()를 사용하여 1차원, 4차원 배열을 동적 할당할 것
		weight_tensor = dmatrix4D(fK, fK, fC_in, fC_out);
		bias_tensor = dmatrix1D(fC_out);


		init(init_type);
	}
	void init(int init_type) {
		// (구현할 것)//////////////////////////////////////////////////////////////////////
		// 동작1: init_type (MEAN_INIT 또는 LOAD_INIT)에 따라 가중치를 다른 방식으로 초기화 함
		// 동작2: MEAN_INIT의 경우 필터는 평균값을 산출하는 필터가 됨 (즉, 모든 가중치 값이 필터의 크기(fK*fK*fC_in)의 역수와 같아짐 (이때 bias는 모두 0으로 설정)
		// 동작3: LOAD_INIT의 경우 filename_weight, filename_bias의 이름을 가지는 파일의 값을 읽어 가중치에 저장(초기화) 함  
		// 함수1: dmatrix4D()와 dmatrix1D()를 사용하여 1차원, 4차원 배열을 동적 할당할 것
		if (init_type == MEAN_INIT) {
			double val = 1.0 / (fK * fK * fC_in);
			for (int y = 0; y < fK; y++) {
				for (int x = 0; x < fK; x++) {
					for (int c = 0; c < fC_in; c++) {
						for (int n = 0; n < fC_out; n++) {
							weight_tensor[y][x][c][n] = val;
						}
					}
				}
			}
			for (int n = 0; n < fC_out; n++) bias_tensor[n] = 0.0;
		}
		else if (init_type == LOAD_INIT) {
		std::ifstream weight_file(filename_weight);
        if (!weight_file.is_open()) {
            // 파일 열기 실패 시 에러 메시지 출력 후 프로그램 종료
            std::cerr << "Error: Could not open weight file: " << filename_weight << std::endl;
            exit(EXIT_FAILURE);
        }

        std::cout << "Loading weights from " << filename_weight << "..." << std::endl;
        for (int y = 0; y < fK; y++) {
            for (int x = 0; x < fK; x++) {
                for (int c = 0; c < fC_in; c++) {
                    for (int n = 0; n < fC_out; n++) {
                        // 파일에서 double 값을 하나씩 읽어와 텐서에 저장
                        weight_file >> weight_tensor[y][x][c][n];
                    }
                }
            }
        }
        weight_file.close(); // 파일 닫기

        // 2. Bias Tensor 불러오기
        std::ifstream bias_file(filename_bias);
        if (!bias_file.is_open()) {
            std::cerr << "Error: Could not open bias file: " << filename_bias << std::endl;
            exit(EXIT_FAILURE);
        }
        
        std::cout << "Loading biases from " << filename_bias << "..." << std::endl;
        for (int n = 0; n < fC_out; n++) {
            bias_file >> bias_tensor[n];
        }
        bias_file.close();}
	}
	~Layer_Conv() override {
		// (구현할 것)//////////////////////////////////////////////////////////////////////
		// 동작1: weight_tensor와 bias_tensor를 동적 할당 해제할 것
		// 함수1: free_dmatrix4D(), free_dmatrix1D() 함수를 사용
		free_dmatrix4D(weight_tensor, fK, fK, fC_in, fC_out);
		free_dmatrix1D(bias_tensor, fC_out);
	}
	Tensor3D* forward(const Tensor3D* input) override {
		// (구현할 것)//////////////////////////////////////////////////////////////////////
		// 동작1: 컨볼루션 (각 위치마다 y = WX + b)를 수행
		// 동작2: output (Tensor3D type)를 먼저 동적 할당하고 연산이 완료된 다음 pointer를 반환 
		int H, W, C;
		input->get_info(H, W, C);
		assert(C == fC_in); // 입력 채널이 일치해야 함

		// 1. 'Same Padding'을 위한 패딩 값 계산
	// 3x3 커널일 경우 pad = 1, 5x5 커널일 경우 pad = 2
		int pad = fK / 2;

		// 2. 출력 텐서의 크기는 입력과 동일하게 설정
		int outH = H;
		int outW = W;

		Tensor3D* output = new Tensor3D(outH, outW, fC_out);

		// 출력 텐서의 모든 픽셀(oh, ow)에 대해 연산
		for (int oh = 0; oh < outH; oh++) {
			for (int ow = 0; ow < outW; ow++) {
				// fC_out 개의 필터(필터 인덱스 f)에 대해 각각 연산
				for (int f = 0; f < fC_out; f++) {
					double sum = 0.0;
					// 커널 윈도우(kh, kw)와 입력 채널(c)에 대해 연산
					for (int kh = 0; kh < fK; kh++) {
						for (int kw = 0; kw < fK; kw++) {
							for (int c = 0; c < fC_in; c++) {
								// 3. 패딩을 고려하여 실제 입력 텐서에서 값을 가져올 위치 계산
								int ih = oh + kh - pad;
								int iw = ow + kw - pad;

								// 4. 경계 검사: 계산된 위치가 입력 텐서의 유효한 범위 내에 있을 때만 값을 읽어옴
								// (범위를 벗어나면 0이 곱해지는 것과 동일)
								if (ih >= 0 && ih < H && iw >= 0 && iw < W) {
									double val = input->get_elem(ih, iw, c);
									sum += val * weight_tensor[kh][kw][c][f];
								}
							}
						}
					}
					sum += bias_tensor[f]; // 바이어스 추가
					output->set_elem(oh, ow, f, sum);
				}
			}
		}
		cout  << name << " is finished" << endl;
		return output;
	};
	void get_info(string& _name, int& _fK, int& _fC_in, int& _fC_out) const override {
		// (구현할 것)//////////////////////////////////////////////////////////////////////
		// 동작: Layer_ReLU와 동일
		_name = name;
		_fK = fK;
		_fC_in = fC_in;
		_fC_out = fC_out;
	}
	void print() const override {
		// (구현할 것)//////////////////////////////////////////////////////////////////////
		// 동작: Layer_ReLU와 동일
		cout << "Layer: " << name << " (Conv) "
			<< "Kernel=" << fK
			<< " Cin=" << fC_in
			<< " Cout=" << fC_out << endl;
	}
};



