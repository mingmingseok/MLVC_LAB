#pragma once
#include "Imagelib.h"
#include "CTensor.h"

#define MEAN_INIT 0
#define LOAD_INIT 1

// Layer�� tensor�� ��/������� ������, Ư�� operation�� �����ϴ� Convolutional Neural Netowork�� �⺻ ���� ����


class Layer {
protected:
	int fK; // kernel size in K*K kernel
	int fC_in; // number of channels
	int fC_out; //number of filters
	string name;
public:
	Layer(string _name, int _fK, int _fC_in, int _fC_out) : name(_name), fK(_fK), fC_in(_fC_in), fC_out(_fC_out) {}
	virtual ~Layer() {}; //����Ҹ��� (����: https://wonjayk.tistory.com/243)
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
		// (������ ��)//////////////////////////////////////////////////////////////////////
		// ����1: Base class�� �����ڸ� ȣ���Ͽ� �ɹ� ������ �ʱ�ȭ �� ��(�ݵ�� initialization list�� ����� ��)
	}
	~Layer_ReLU() {}
	Tensor3D* forward(const Tensor3D* input) override {
		// (������ ��)
		// ����1: input tensor�� ���� �� element x�� ����̸� �״�� ����, �����̸� 0���� output tensor�� �����Ұ�    
		// ����2: �̶�, output tensor�� �����Ҵ��Ͽ� �ּҰ��� ��ȯ�� ��
		// �Լ�1: Tensor3D�� �ɹ��Լ��� get_info(), get_elem(), set_elem()�� ������ Ȱ���� ��
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
		// (������ ��)//////////////////////////////////////////////////////////////////////
		// ����: Tensor3D�� get_info()�� ���������� �ɹ� �������� pass by reference�� �ܺο� ����
		_name = name;
		_fK = fK;
		_fC_in = fC_in;
		_fC_out = fC_out;
	}
	void print() const override {
		// (������ ��)//////////////////////////////////////////////////////////////////////
		// ����: Tensor3D�� print()�� ���������� ������ ũ�⸦ ȭ�鿡 ���
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
	double**** weight_tensor; // fK x fK x _fC_in x _fC_out ũ�⸦ ������ 4���� �迭
	double*  bias_tensor;     // _fC_out ũ�⸦ ������ 1���� �迭 (bias�� �� filter�� 1�� ����) 
public:
	Layer_Conv(string _name, int _fK, int _fC_in, int _fC_out, int init_type, string _filename_weight = "", string _filename_bias = "")
	:Layer(_name, _fK, _fC_in, _fC_out), 
          filename_weight(_filename_weight), filename_bias(_filename_bias)
	{
		// (������ ��)//////////////////////////////////////////////////////////////////////
		// ����1: initialization list�� base class�� �����ڸ� �̿��Ͽ� �ɹ� ������ �ʱ�ȭ �� ��
		// ����2: filename_weight�� filename_bias�� LOAD_INIT ����� ��� �ش� ���Ϸκ��� ����ġ/���̾�� �ҷ���
		// ����3: init() �Լ��� init_type�� �Է����� �޾� ����ġ�� �ʱ�ȭ �� 
		// �Լ�1: dmatrix4D()�� dmatrix1D()�� ����Ͽ� 1����, 4���� �迭�� ���� �Ҵ��� ��
		weight_tensor = dmatrix4D(fK, fK, fC_in, fC_out);
		bias_tensor = dmatrix1D(fC_out);


		init(init_type);
	}
	void init(int init_type) {
		// (������ ��)//////////////////////////////////////////////////////////////////////
		// ����1: init_type (MEAN_INIT �Ǵ� LOAD_INIT)�� ���� ����ġ�� �ٸ� ������� �ʱ�ȭ ��
		// ����2: MEAN_INIT�� ��� ���ʹ� ��հ��� �����ϴ� ���Ͱ� �� (��, ��� ����ġ ���� ������ ũ��(fK*fK*fC_in)�� ������ ������ (�̶� bias�� ��� 0���� ����)
		// ����3: LOAD_INIT�� ��� filename_weight, filename_bias�� �̸��� ������ ������ ���� �о� ����ġ�� ����(�ʱ�ȭ) ��  
		// �Լ�1: dmatrix4D()�� dmatrix1D()�� ����Ͽ� 1����, 4���� �迭�� ���� �Ҵ��� ��
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
            // ���� ���� ���� �� ���� �޽��� ��� �� ���α׷� ����
            std::cerr << "Error: Could not open weight file: " << filename_weight << std::endl;
            exit(EXIT_FAILURE);
        }

        std::cout << "Loading weights from " << filename_weight << "..." << std::endl;
        for (int y = 0; y < fK; y++) {
            for (int x = 0; x < fK; x++) {
                for (int c = 0; c < fC_in; c++) {
                    for (int n = 0; n < fC_out; n++) {
                        // ���Ͽ��� double ���� �ϳ��� �о�� �ټ��� ����
                        weight_file >> weight_tensor[y][x][c][n];
                    }
                }
            }
        }
        weight_file.close(); // ���� �ݱ�

        // 2. Bias Tensor �ҷ�����
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
		// (������ ��)//////////////////////////////////////////////////////////////////////
		// ����1: weight_tensor�� bias_tensor�� ���� �Ҵ� ������ ��
		// �Լ�1: free_dmatrix4D(), free_dmatrix1D() �Լ��� ���
		free_dmatrix4D(weight_tensor, fK, fK, fC_in, fC_out);
		free_dmatrix1D(bias_tensor, fC_out);
	}
	Tensor3D* forward(const Tensor3D* input) override {
		// (������ ��)//////////////////////////////////////////////////////////////////////
		// ����1: ������� (�� ��ġ���� y = WX + b)�� ����
		// ����2: output (Tensor3D type)�� ���� ���� �Ҵ��ϰ� ������ �Ϸ�� ���� pointer�� ��ȯ 
		int H, W, C;
		input->get_info(H, W, C);
		assert(C == fC_in); // �Է� ä���� ��ġ�ؾ� ��

		// 1. 'Same Padding'�� ���� �е� �� ���
	// 3x3 Ŀ���� ��� pad = 1, 5x5 Ŀ���� ��� pad = 2
		int pad = fK / 2;

		// 2. ��� �ټ��� ũ��� �Է°� �����ϰ� ����
		int outH = H;
		int outW = W;

		Tensor3D* output = new Tensor3D(outH, outW, fC_out);

		// ��� �ټ��� ��� �ȼ�(oh, ow)�� ���� ����
		for (int oh = 0; oh < outH; oh++) {
			for (int ow = 0; ow < outW; ow++) {
				// fC_out ���� ����(���� �ε��� f)�� ���� ���� ����
				for (int f = 0; f < fC_out; f++) {
					double sum = 0.0;
					// Ŀ�� ������(kh, kw)�� �Է� ä��(c)�� ���� ����
					for (int kh = 0; kh < fK; kh++) {
						for (int kw = 0; kw < fK; kw++) {
							for (int c = 0; c < fC_in; c++) {
								// 3. �е��� ����Ͽ� ���� �Է� �ټ����� ���� ������ ��ġ ���
								int ih = oh + kh - pad;
								int iw = ow + kw - pad;

								// 4. ��� �˻�: ���� ��ġ�� �Է� �ټ��� ��ȿ�� ���� ���� ���� ���� ���� �о��
								// (������ ����� 0�� �������� �Ͱ� ����)
								if (ih >= 0 && ih < H && iw >= 0 && iw < W) {
									double val = input->get_elem(ih, iw, c);
									sum += val * weight_tensor[kh][kw][c][f];
								}
							}
						}
					}
					sum += bias_tensor[f]; // ���̾ �߰�
					output->set_elem(oh, ow, f, sum);
				}
			}
		}
		cout  << name << " is finished" << endl;
		return output;
	};
	void get_info(string& _name, int& _fK, int& _fC_in, int& _fC_out) const override {
		// (������ ��)//////////////////////////////////////////////////////////////////////
		// ����: Layer_ReLU�� ����
		_name = name;
		_fK = fK;
		_fC_in = fC_in;
		_fC_out = fC_out;
	}
	void print() const override {
		// (������ ��)//////////////////////////////////////////////////////////////////////
		// ����: Layer_ReLU�� ����
		cout << "Layer: " << name << " (Conv) "
			<< "Kernel=" << fK
			<< " Cin=" << fC_in
			<< " Cout=" << fC_out << endl;
	}
};



