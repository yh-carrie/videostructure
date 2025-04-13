for i in range(attns.shape[0]):  # 遍历每个样本
    attn_map = attns[i].cpu().numpy()  # 获取注意力图并转换为 NumPy 数组
    attn_map = attn_map[1:, :]
    attn_map = attn_map.reshape(18, 9)
    attn_map = cv2.resize(attn_map, (imgs.shape[3], imgs.shape[2]), interpolation=cv2.INTER_LINEAR)  # 调整大小与原图一致
    attn_map = (attn_map - attn_map.min()) / (attn_map.max() - attn_map.min())  # 归一化
    attn_map = plt.cm.jet(attn_map)[:, :, :3]  # 应用颜色映射
    attn_map = attn_map.astype(np.float32)

    # 将热力图叠加到原图像
    original_img = imgs[i].cpu().numpy().transpose(1, 2, 0)  # 转换为 HWC 格式
    original_img = (original_img - original_img.min()) / (original_img.max() - original_img.min())  # 归一化

    # 打印形状以进行调试
    print(f"Original Image Shape: {original_img.shape}")
    print(f"Attention Map Shape: {attn_map.shape}")

    # 确保两个图像的形状一致
    if original_img.shape[:2] != attn_map.shape[:2]:
        attn_map = cv2.resize(attn_map, (original_img.shape[1], original_img.shape[0]))

    overlayed_img = cv2.addWeighted(original_img, 0.6, attn_map, 0.4, 0)  # 叠加热力图

    # 保存热力图
    plt.figure(figsize=(10, 5))  # 设置图形大小
    plt.imshow(overlayed_img)
    plt.axis('off')
    plt.savefig(f'heatmap_{imgname[i]}.png', bbox_inches='tight')
    plt.close()