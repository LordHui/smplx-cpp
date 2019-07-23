#include <iostream>
#include <fstream>
#include <Eigen/Eigen>

#include "smplx.h"
#include "math_util.h"


Smplx::Conf::Conf(const std::string& confFile)
{
	std::ifstream ifs(confFile);
	if (!ifs.is_open()) {
		std::cerr << "cannot open " << confFile << std::endl;
		exit(-1);
	}
	std::string str;
	ifs >> str >> jointSize;
	ifs >> str >> somatotypeSize;
	ifs >> str >> expressionSize;
	ifs >> str >> faceSize;
	ifs >> str >> vertexSize;
	shapeSize = somatotypeSize + expressionSize;
	ifs.close();
}


Smplx::Param::Param(const Conf& _conf)
	:conf(_conf)
{
	data = Eigen::VectorXf::Zero(3 + conf.jointSize * 3 + conf.somatotypeSize + conf.expressionSize);
}


Smplx::Smplx(const std::string &modelPath)
	:conf(modelPath + "/conf.txt")
{
	std::ifstream ifs;

	// parents
	ifs.open(modelPath + "/parent.txt");
	m_parent.resize(conf.jointSize);
	for (int jIdx = 0; jIdx < conf.jointSize; jIdx++)
		ifs >> m_parent(jIdx);
	ifs.close();

	// vertices
	ifs.open(modelPath + "/vertices.txt");
	m_vertices.resize(3, conf.vertexSize);
	for (int vIdx = 0; vIdx < conf.vertexSize; vIdx++)
		for (int i = 0; i < 3; i++)
			ifs >> m_vertices(i, vIdx);
	ifs.close();

	// faces
	ifs.open(modelPath + "/faces.txt");
	m_faces.resize(3, conf.faceSize);
	for (int fIdx = 0; fIdx < conf.faceSize; fIdx++)
		for (int i = 0; i < 3; i++)
			ifs >> m_faces(i, fIdx);
	ifs.close();

	// jregressor (load in transpose)
	ifs.open(modelPath + "/jregressor.txt");
	m_jRegressor.resize(conf.vertexSize, conf.jointSize);
	for (int jIdx = 0; jIdx < conf.jointSize; jIdx++)
		for (int vIdx = 0; vIdx < conf.vertexSize; vIdx++)
			ifs >> m_jRegressor(vIdx, jIdx);
	ifs.close();

	// lbsweight (load in transpose)
	ifs.open(modelPath + "/lbs_weights.txt");
	m_lbsWeights.resize(conf.jointSize, conf.vertexSize);
	for (int vIdx = 0; vIdx < conf.vertexSize; vIdx++)
		for (int jIdx = 0; jIdx < conf.jointSize; jIdx++)
			ifs >> m_lbsWeights(jIdx, vIdx);
	ifs.close();

	// vshapeblend
	ifs.open(modelPath + "/shape_blend.txt");
	m_vShapeblend.resize(conf.vertexSize * 3, conf.shapeSize);
	for (int row = 0; row < m_vShapeblend.rows(); row++)
		for (int col = 0; col < m_vShapeblend.cols(); col++)
			ifs >> m_vShapeblend(row, col);
	ifs.close();

	//// vposeblend (load in transpose)
	//ifs.open(modelPath + "/pose_blend.txt");
	//m_vPoseblend.resize(conf.vertexSize * 3, 9 * (conf.jointSize - 1));
	//for (int col = 0; col < m_vPoseblend.cols(); col++)
	//	for (int row = 0; row < m_vPoseblend.rows(); row++)
	//		ifs >> m_vPoseblend(row, col);
	//ifs.close();

	// joint
	m_joints = m_vertices * m_jRegressor;

	// jshapeblend
	m_jShapeblend.resize(3 * conf.jointSize, conf.shapeSize);
	for (int shapeIdx = 0; shapeIdx < m_jShapeblend.cols(); shapeIdx++) {
		Eigen::Map<Eigen::MatrixXf>(m_jShapeblend.col(shapeIdx).data(), 3, conf.jointSize)
			= Eigen::Map<Eigen::MatrixXf>(m_vShapeblend.col(shapeIdx).data(), 3, conf.vertexSize) * m_jRegressor;
	}
}


void Smplx::CalcJBlend(const Smplx::Param& param, const int& jMask, Eigen::Matrix3Xf* jBlend) const
{
	Eigen::VectorXf jOffset = m_jShapeblend.block(0, 0, 3 * jMask, conf.shapeSize) * param.GetShape();
	*jBlend = m_joints.block(0, 0, 3, jMask) + Eigen::Map<Eigen::MatrixXf>(jOffset.data(), 3, jMask);
}


void  Smplx::CalcNodeWarps(const Param& param, const int& jMask, Eigen::Matrix4Xf* nodeWarps) const
{
	Eigen::Matrix3Xf jBlend;
	CalcJBlend(param, jMask, &jBlend);

	nodeWarps->resize(4, 4 * jMask);
	for (int jIdx = 0; jIdx < jMask; jIdx++) {
		Eigen::Matrix4f matrix = Eigen::Matrix4f::Identity();
		matrix.topLeftCorner(3, 3) = MathUtil::Rodrigues(param.GetPose(jIdx));

		if (jIdx == 0)
			matrix.topRightCorner(3, 1) = jBlend.col(jIdx) + param.GetTrans();
		else
			matrix.topRightCorner(3, 1) = jBlend.col(jIdx) - jBlend.col(m_parent(jIdx));

		nodeWarps->block<4, 4>(0, 4 * jIdx) = matrix;
	}
}


void Smplx::CalcChainWarps(const Param& param, const int& jMask, Eigen::Matrix4Xf* chainWarps, Eigen::Matrix4Xf* nodeWarps) const
{
	Eigen::Matrix4Xf tmpNodeWarps;
	nodeWarps = nodeWarps == nullptr ? &tmpNodeWarps : nodeWarps;

	CalcNodeWarps(param, jMask, nodeWarps);
	chainWarps->resize(4, 4 * jMask);
	for (int jIdx = 0; jIdx < jMask; jIdx++)
		if (jIdx == 0)
			chainWarps->block<4, 4>(0, 4 * jIdx) = nodeWarps->block<4, 4>(0, 4 * jIdx);
		else
			chainWarps->block<4, 4>(0, 4 * jIdx) = chainWarps->block<4, 4>(0, 4 * m_parent(jIdx))*nodeWarps->block<4, 4>(0, 4 * jIdx);
}


void Smplx::CalcJFinal(const Param& param, const int& jMask, Eigen::Matrix3Xf* jFinal, Eigen::Matrix4Xf* chainWarps, Eigen::Matrix4Xf* nodeWarps) const
{
	Eigen::Matrix4Xf tmpChainWarps, tmpNodeWarps;
	chainWarps = chainWarps == nullptr ? &tmpChainWarps : chainWarps;
	nodeWarps = nodeWarps == nullptr ? &tmpNodeWarps : nodeWarps;

	CalcChainWarps(param, jMask, chainWarps, nodeWarps);

	jFinal->resize(3, jMask);
	for (int jIdx = 0; jIdx < jFinal->cols(); jIdx++)
		jFinal->col(jIdx) = chainWarps->block<3, 1>(0, 4 * jIdx + 3);
}


void Smplx::CalcVBlend(const Param& param, Eigen::Matrix3Xf* vBlend) const
{
	Eigen::VectorXf vShapeOffset = m_vShapeblend * param.GetShape();
	*vBlend = m_vertices + Eigen::Map<Eigen::MatrixXf>(vShapeOffset.data(), 3, conf.vertexSize);

	//Eigen::Matrix3Xf poseFeature(3, 3 * (conf.jointSize - 1));
	//for (int jIdx = 1; jIdx < conf.jointSize; jIdx++)
	//	poseFeature.block<3, 3>(0, (jIdx - 1) * 3) = MathUtil::Rodrigues(param.GetPose(jIdx)).transpose() - Eigen::Matrix3f::Identity();
	//Eigen::VectorXf vPoseOffset = m_vPoseblend * Eigen::Map<Eigen::VectorXf>(poseFeature.data(), 9 * (conf.jointSize - 1));
	//
	//*vBlend = m_vertices + Eigen::Map<Eigen::MatrixXf>(vShapeOffset.data(), 3, conf.vertexSize) + Eigen::Map<Eigen::MatrixXf>(vPoseOffset.data(), 3, conf.vertexSize);
}


void Smplx::CalcVFinal(const Param& param, Eigen::Matrix3Xf* vFinal) const
{
	Eigen::Matrix3Xf jBlend, vBlend;
	CalcJBlend(param, conf.jointSize, &jBlend);
	CalcVBlend(param, &vBlend);

	Eigen::Matrix4Xf chainWarps;
	CalcChainWarps(param, conf.jointSize, &chainWarps);

	Eigen::Matrix4Xf chainWarpsNormalized = chainWarps;
	for (int jIdx = 0; jIdx < conf.jointSize; jIdx++)
		chainWarpsNormalized.block<3, 1>(0, jIdx * 4 + 3) -= (chainWarps.block<3, 3>(0, jIdx * 4)*jBlend.col(jIdx));

	vFinal->resize(3, conf.vertexSize);
	for (int vIdx = 0; vIdx < conf.vertexSize; vIdx++) {
		Eigen::Matrix4f warp;
		Eigen::Map<Eigen::VectorXf>(warp.data(), 16)
			= Eigen::Map<Eigen::MatrixXf>(chainWarpsNormalized.data(), 16, conf.jointSize) * (m_lbsWeights.col(vIdx));
		vFinal->col(vIdx) = warp.topLeftCorner(3, 4)*(vBlend.col(vIdx).homogeneous());
	}
}


void Smplx::Debug(const Param& param, const std::string& filename) const
{
	Eigen::Matrix3Xf vFinal;
	CalcVFinal(param, &vFinal);

	std::ofstream fs(filename);
	for (int i = 0; i < vFinal.cols(); i++)
		fs << "v " << vFinal(0, i) << " " << vFinal(1, i) << " " << vFinal(2, i) << std::endl;

	for (int i = 0; i < m_faces.cols(); i++)
		fs << "f " << m_faces(0, i) + 1 << " " << m_faces(1, i) + 1 << " " << m_faces(2, i) + 1 << std::endl;

	fs.close();
}
