#pragma once
#include <Eigen/Core>
#include <string>


class Smplx
{
public:
	const static int JOINT_SIZE = 55;
	const static int SHAPE_SIZE = 10;
	const static int EXPRESSION_SIZE = 10;
	const static int SHAPE_EXPRESSION_SIZE = SHAPE_SIZE + EXPRESSION_SIZE;


	struct Param
	{
		Eigen::Vector3f translation;
		Eigen::VectorXf pose;
		Eigen::VectorXf shape;
		Eigen::VectorXf expression;

		Param();
		void SetZero();
		void SetRandom();
	};

	Smplx(const std::string &modelPath);
	~Smplx() = default;
	Smplx(const Smplx& _) = delete;
	Smplx& operator=(const Smplx& _) = delete;

	void CalcJBlend(const Param& param, const int& jMask, Eigen::Matrix3Xf* jBlend) const;
	void CalcNodeWarps(const Param& param, const int& jMask, Eigen::Matrix4Xf* nodeWarps) const;
	void CalcChainWarps(const Param& param, const int& jMask, Eigen::Matrix4Xf* chainWarps, Eigen::Matrix4Xf* nodeWarps = nullptr) const;
	void CalcJFinal(const Param& param, const int& jMask, Eigen::Matrix3Xf* jFinal, Eigen::Matrix4Xf* chainWarps = nullptr, Eigen::Matrix4Xf* nodeWarps = nullptr) const;
	void CalcVBlend(const Param& param, Eigen::Matrix3Xf* vBlend) const;
	void CalcVFinal(const Param& param, Eigen::Matrix3Xf* vFinal) const;
	void Debug(const Param& param, const std::string& filename) const;

protected:
	int m_faceSize;
	int m_vertexSize;

	Eigen::Matrix3Xf m_joints;
	Eigen::Matrix3Xf m_vertices;
	Eigen::Matrix3Xi m_faces;
	Eigen::VectorXi m_parent;
	Eigen::MatrixXf m_jRegressor;
	Eigen::MatrixXf m_lbsWeights;
	Eigen::MatrixXf m_vShapeblend;
	Eigen::MatrixXf m_vPoseblend;
	Eigen::MatrixXf m_jShapeblend;
};
