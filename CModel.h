#pragma once
#include "CLayer.h"

// Model�� layer�� tensor���� ��� ���� �����Ͽ� ȿ�������� CNN�� ����� �� �ֵ��� ��

class Model {
private:
	vector<Layer*> layers; //layer���� ���������� ����
	vector<Tensor3D*> tensors;// tensor���� ���������� ���� ( 0��° tensor�� 0��° layer�� �Է�, ���������� 1��° tensor�� 1��° layer�� �Է����� 0��° layer�� �����)
public:
	Model() {}
	void add_layer(Layer* layer) {
		// (������ ��) //////////////////////////////////////////////////
		// ����: layer ��ü�� layers vector�� ������ element�� ���� 
		layers.push_back(layer);
	}
	~Model() {
		// (������ ��)//////////////////////////////////////////////////
		// ����: layers�� tensors�� ��� element�� �����Ҵ� ������ �� ��
		for (auto& l : layers) {
			delete l;
		}
		layers.clear();

		for (auto& t : tensors) {
			delete t;
		}
		tensors.clear();
	}
	void test(string filename_input, string filename_output) {
		// (������ ��)//////////////////////////////////////////////////
		// ����1: filename_input���κ��� �̹����� �о�ͼ�, tensor�� ��ȯ�� ���� CNN�� �����Ѵ��� �� ������� filename_output�� ����
		// ����2: �ּ� (1), (2), (3), (4) �� (2)���� �����ϸ� �� 

		int nH, nW;
		double** input_img_Y, **input_img_U, **input_img_V;
		byte* pLoadImage;

		// (1) ������ �о 2���� �迭�� ���� (input_img_Y, U, V�� read_image���� ���� �Ҵ��)
		read_image(filename_input, pLoadImage, input_img_Y, input_img_U, input_img_V, nH, nW);
		cout << "Reading (" << filename_input << ") is complete..." << endl;


		// (2) �̺κи� ������ ��//////////////////////////////////////////////////
		// ����1: ���� tensors�� 0��° element�� ����(CNN�� �Է�)�� �̹� ����Ǿ� ����
		// ����2: tensors vector�� i��° tensor�� layers vector�� �ִ� i��° layer�� forward�Լ��� �Է¹ް�, �� ����� tensors vector�� i+1��° tensor�� ������
		// ����3: ��������� tensors�� ���� ������ tensor�� CNN�� ��°��� �� (�� ��°��� (3)���� 1���� �迭�� ��ȯ�Ǿ� �̹��� ���Ͽ� ����� 
		for (size_t i = 0; i < layers.size(); i++) {
			Tensor3D* input_tensor = tensors.at(i);            // i��° tensor�� �Է�����
			Tensor3D* output_tensor = layers.at(i)->forward(input_tensor); // layer forward
			tensors.push_back(output_tensor);                  // i+1��° tensor�� �߰�

		}
		Tensor3D* input_tensor = tensors.at(0);
		Tensor3D* residual_tensor = tensors.at(tensors.size() - 1);

		int H, W, C;
		input_tensor->get_info(H, W, C);

		// ���� ����� ������ ���ο� �ټ��� ����
		Tensor3D* final_image_tensor = new Tensor3D(H, W, C);

		for (int h = 0; h < H; h++) {
			for (int w = 0; w < W; w++) {
				double input_val = input_tensor->get_elem(h, w, 0);
				double residual_val = residual_tensor->get_elem(h, w, 0);
				// ���� + ���� �� = ���� ���
				final_image_tensor->set_elem(h, w, 0, input_val + residual_val);
			}	
		}
		// �ϼ��� ���� �̹����� �ټ� ����� �� �ڿ� �߰�
		tensors.push_back(final_image_tensor);

		cout << "Super-resolution is complete..." << endl;

		cout << "2";




		// (3) CNN�� ���(������ tensor)�� 2���� �迭�� ��ȯ �� U, V ä�ΰ� �Բ� �̹����� ����
		Tensor3D* output_tensor_Y = tensors.at(tensors.size() - 1);
		output_tensor_Y->print();
		save_image(filename_output, pLoadImage, output_tensor_Y, input_img_U, input_img_V, nH, nW);
		cout << "Saving (" << filename_output << ") is complete..." << endl;

		cout << "3";

		// (4) �Ҵ� ����
		free(pLoadImage);
		free_dmatrix2D(input_img_Y, nH, nW);
		free_dmatrix2D(input_img_U, nH, nW);
		free_dmatrix2D(input_img_V, nH, nW);

		cout << "4";
	}

	void read_image(const string filename, byte*& pLoadImage, double**& img_Y, double**& img_U, double**& img_V, int& nH, int& nW) {

		LoadBmp(filename.c_str(), &pLoadImage, nH, nW);///�̹������� �б�

		img_Y = dmatrix2D(nH, nW);
		img_U = dmatrix2D(nH, nW);
		img_V = dmatrix2D(nH, nW);

		convert1Dto2D(pLoadImage, img_Y, img_U, img_V, nH, nW);

		// �Է� ������ tensor�� ��ȯ �� ù��° element�� ����
		double*** inImage3D = dmatrix3D(nH, nW, 1);
		convert2Dto3D(img_Y, inImage3D, nH, nW);

		Tensor3D* temp = new Tensor3D(nH, nW, 1);
		temp->set_tensor(inImage3D);
		tensors.push_back(temp);

	}
	void save_image(string filename, byte*& pLoadImage, Tensor3D*& tensor_Y, double** img_U, double** img_V, int nH, int nW) {
		double** img_Y = dmatrix2D(nH, nW);
		convert3Dto2D(tensor_Y->get_tensor(), img_Y, nH, nW);
		convert2Dto1D(img_Y, img_U, img_V, pLoadImage, nH, nW);
		SaveBmp(filename.c_str(), pLoadImage, nH, nW);
		free_dmatrix2D(img_Y, nH, nW);
	}
	void print_layer_info() const {
		cout << endl << "(Layer information)_____________" << endl;
		for (unsigned i = 0; i < layers.size(); i++) {
			cout << i + 1 << "-th layer: ";
			layers.at(i)->print();
		}
	}
	void print_tensor_info() const {
		cout << endl << "(Tensor information)_____________" << endl;
		for (unsigned i = 0; i < tensors.size(); i++) {
			cout << i + 1 << "-th tensor: ";
			tensors.at(i)->print();
		}
	}

	//	void train();
};