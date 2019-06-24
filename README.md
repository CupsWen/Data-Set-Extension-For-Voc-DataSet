# Data-Set-Extension-For-Voc-DataSet

对VOC格式组织的数据集进行翻转、平移、缩放、添加噪声、过曝光、欠曝光等操作进行1--->28的数据集扩充。
因为VOC的boundingbox一旦旋转，很可能盖不住，所以没有添加。

voc  1--->28

第一组扩展	(该类自带)
原图片 ---> 	_flip_x
			_flip_y
			_flip_x_y

第二组扩展	gaussian_blur_fun
第一组 --->	第一组全体高斯模糊

第三组扩展	change_exposure_fun
第一组 --->	第一组全体进行曝光、和欠曝光

第三组扩展	add_salt_noise
第一组 --->  第一组全体加入椒盐噪声

第四组扩展	add_Translate
第一组 --->	第一组进行平移

第五组扩展	add_Resize
第一组 -->	第一组进行缩放
