#pragma once
#include <Eigen/Core>
#include <string>


class Smplx
{
public:
	struct Conf
	{
		int jointSize;
		int somatotypeSize;
		int expressionSize;
		int shapeSize;
		int faceSize;
		int vertexSize;
		Conf(const std::string& confFile);
	};

	struct Param
	{
		const Conf conf;
		Eigen::VectorXf data;

		auto GetTrans() { return data.segment<3>(0); }
		auto GetTrans() const { return data.segment<3>(0); }

		auto GetPose() { return data.segment(3, conf.jointSize * 3); }
		auto GetPose() const { return data.segment(3, conf.jointSize * 3); }

		auto GetPose(const int& jIdx) { return data.segment<3>(3 + jIdx * 3); }
		auto GetPose(const int& jIdx) const { return data.segment<3>(3 + jIdx * 3); }

		auto GetTransPose() { return data.segment(0, 3 + conf.jointSize * 3); }
		auto GetTransPose() const { return data.segment(0, 3 + conf.jointSize * 3); }

		auto GetSomatotype() { return data.segment(3 + conf.jointSize * 3, conf.somatotypeSize); }
		auto GetSomatotype() const { return data.segment(3 + conf.jointSize * 3, conf.somatotypeSize); }

		auto GetExpression() { return data.segment(3 + conf.jointSize * 3 + conf.somatotypeSize, conf.expressionSize); }
		auto GetExpression() const { return data.segment(3 + conf.jointSize * 3 + conf.somatotypeSize, conf.expressionSize); }

		auto GetShape() { return data.segment(3 + conf.jointSize * 3, conf.somatotypeSize + conf.expressionSize); }
		auto GetShape() const { return data.segment(3 + conf.jointSize * 3, conf.somatotypeSize + conf.expressionSize); }

		Param(const Conf& _conf);
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

	const Conf conf;
protected:
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
