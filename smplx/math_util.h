#pragma once

#include <cmath>
#include <Eigen/Eigen>


namespace MathUtil
{
	template<typename T>
	inline T Max(const T& a, const T& b)
	{
		return a > b ? a : b;
	}


	template<typename T>
	inline T Min(const T& a, const T& b)
	{
		return a < b ? a : b;
	}


	template<typename T>
	inline T Trunc(const T& x, const T& min, const T& max)
	{
		return Max(min, Min(x, max));
	}


	inline Eigen::Matrix3f Rodrigues(const Eigen::Vector3f& vec)
	{
		const float angle = vec.norm();
		if (angle < 1e-5f)
			return Eigen::Matrix3f::Identity();
		else
			return  Eigen::AngleAxisf(angle, vec / angle).matrix();
	}


	inline Eigen::Vector3f InvRodrigues(const Eigen::Matrix3f& mat)
	{
		const Eigen::AngleAxisf angleAxis(mat);
		return angleAxis.angle() * angleAxis.axis();
	}


	inline Eigen::Matrix3f Skew(const Eigen::Vector3f& vec)
	{
		Eigen::Matrix3f skew;
		skew << 0.f, -vec.z(), vec.y(),
			vec.z(), 0.f, -vec.x(),
			-vec.y(), vec.x(), 0.f;
		return skew;
	}
}


