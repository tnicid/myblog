---


---

<h4 id="lsp数据集简介">LSP数据集简介</h4>
<ul>
<li>LSP &amp;&amp; LSP_extended<br>
这个数据集是由Flickr上‘Volleyball’, ‘Badminton’, ‘Athletics’, ‘Baseball’,  ‘Gymnastics’, ‘Parkour’, ‘Soccer’, ‘Tennis’（原数据集）, ‘parkour’, ‘gymnastics’, and ‘athletics’ （扩展集）等标签所构成。每个图片都由 Amazon Mechanical Turk和类似的途径标注而来，并不高度准确。这些图片被缩放至每个人大约150px长度进行标注，包含了14个节点。<br>
LSP地址：<a href="http://sam.johnson.io/research/lsp_dataset.zip">http://sam.johnson.io/research/lsp_dataset.zip</a><br>
LSP样本数：2000个（全身，单人）<br>
LSP_extended地址：<a href="http://sam.johnson.io/research/lspet_dataset.zip">http://sam.johnson.io/research/lspet_dataset.zip</a><br>
LSP_extended样本数：10000个（全身，单人）</li>
<li>LSP &amp;&amp; LSP_extended 共12000个标注，节点是以下十四个：<br>
{1. Right ankle 2. Right knee 3. Right hip 4. Left hip 5. Left knee 6.Left ankle 7.Right wrist <br> 8. Right elbow 9. Right shoulder 10. Left shoulder 11. Left elbow 12. Left wrist 13. Neck 14. Head top}<br><br>
由于是单人数据集，该数据集的训练难度比多人数据集更简单。</li>
</ul>
<h4 id="mpii数据集简介">MPII数据集简介</h4>
<p>MPII人体姿势数据集是人体姿势预估的一个benchmark，数据集包括了超过40k人的25000张带标注图片，这些图片是从YouTube video中抽取出来的。在测试集中还收录了身体部位遮挡、3D躯干、头部方向的标注。<br>
MPII地址：<a href="http://human-pose.mpi-inf.mpg.de/#overview">http://human-pose.mpi-inf.mpg.de/#overview</a><br>
MPII样本数：25000（单人、多人）<br>
包括以下16类标注： {Head – 0, Neck – 1, Right Shoulder – 2, Right Elbow – 3, Right Wrist – 4,  Left Shoulder – 5, Left Elbow – 6, Left Wrist – 7, Right Hip – 8,  Right Knee – 9, Right Ankle – 10, Left Hip – 11, Left Knee – 12,  Left Ankle – 13, Chest – 14, Background – 15}</p>
<h4 id="数据集处理">数据集处理</h4>
<ol>
<li>matlab格式读入：<br>
文件joints.mat是MATLAB数据格式，包含了一个以x坐标、y坐标和一个表示关节可见性的二进制数字所构成的3x14x10000的矩阵。<br>
使用模块scipy.io的函数loadmat和savemat可以实现对mat数据的读写</li>
</ol>
<pre class=" language-python"><code class="prism  language-python"><span class="token keyword">import</span> scipy<span class="token punctuation">.</span>io <span class="token keyword">as</span> sio
<span class="token keyword">import</span> numpy <span class="token keyword">as</span> np
data <span class="token operator">=</span> sio<span class="token punctuation">.</span>loadmat<span class="token punctuation">(</span>self<span class="token punctuation">.</span>lsp_anno_path<span class="token punctuation">[</span>count<span class="token punctuation">]</span><span class="token punctuation">)</span>
joints <span class="token operator">=</span> data<span class="token punctuation">[</span><span class="token string">'joints'</span><span class="token punctuation">]</span>
</code></pre>
<ol start="2">
<li>2 json格式读入<br>
MPII数据集是以json的格式进行的标注，可以通过json库进行读入</li>
</ol>
<pre class=" language-python"><code class="prism  language-python"><span class="token keyword">import</span> json
anno <span class="token operator">=</span> json<span class="token punctuation">.</span>load<span class="token punctuation">(</span>self<span class="token punctuation">.</span>mpii_anno_pah<span class="token punctuation">)</span>
</code></pre>
<ol start="4">
<li>将每个图片打包成（图片，标注，bounding box）的形式，bounding box即图片大小，其目的是将大小不一的图片处理成256 x 256的大小：</li>
</ol>
<pre class=" language-python"><code class="prism  language-python"><span class="token keyword">from</span> PIL <span class="token keyword">import</span> Image
<span class="token keyword">for</span> idd<span class="token punctuation">,</span> joint_idd <span class="token keyword">in</span> <span class="token builtin">enumerate</span><span class="token punctuation">(</span>joints<span class="token punctuation">)</span><span class="token punctuation">:</span>
    image_name <span class="token operator">=</span> <span class="token string">"im%s.jpg"</span> <span class="token operator">%</span> <span class="token builtin">str</span><span class="token punctuation">(</span>idd <span class="token operator">+</span> <span class="token number">1</span><span class="token punctuation">)</span><span class="token punctuation">.</span>zfill<span class="token punctuation">(</span><span class="token number">5</span><span class="token punctuation">)</span> <span class="token keyword">if</span> count <span class="token keyword">else</span> <span class="token string">"im%s.jpg"</span> <span class="token operator">%</span> <span class="token builtin">str</span><span class="token punctuation">(</span>idd <span class="token operator">+</span> <span class="token number">1</span><span class="token punctuation">)</span><span class="token punctuation">.</span>zfill<span class="token punctuation">(</span><span class="token number">4</span><span class="token punctuation">)</span>
    joint_id <span class="token operator">=</span> idd <span class="token operator">+</span> <span class="token builtin">len</span><span class="token punctuation">(</span>joints<span class="token punctuation">)</span> <span class="token keyword">if</span> count <span class="token keyword">else</span> idd
    im_path <span class="token operator">=</span> os<span class="token punctuation">.</span>path<span class="token punctuation">.</span>join<span class="token punctuation">(</span>self<span class="token punctuation">.</span>lsp_data_path<span class="token punctuation">[</span>count<span class="token punctuation">]</span><span class="token punctuation">,</span> image_name<span class="token punctuation">)</span>
    im <span class="token operator">=</span> Image<span class="token punctuation">.</span><span class="token builtin">open</span><span class="token punctuation">(</span>im_path<span class="token punctuation">)</span>
    im <span class="token operator">=</span> np<span class="token punctuation">.</span>asarray<span class="token punctuation">(</span>im<span class="token punctuation">)</span>
    shape <span class="token operator">=</span> im<span class="token punctuation">.</span>shape
    bbox <span class="token operator">=</span> <span class="token punctuation">[</span><span class="token number">0</span><span class="token punctuation">,</span> <span class="token number">0</span><span class="token punctuation">,</span> shape<span class="token punctuation">[</span><span class="token number">1</span><span class="token punctuation">]</span><span class="token punctuation">,</span> shape<span class="token punctuation">[</span><span class="token number">0</span><span class="token punctuation">]</span><span class="token punctuation">]</span>
    joint_dict<span class="token punctuation">[</span>joint_id<span class="token punctuation">]</span> <span class="token operator">=</span> <span class="token punctuation">{</span><span class="token string">'imgpath'</span><span class="token punctuation">:</span> im_path<span class="token punctuation">,</span> <span class="token string">'joints'</span><span class="token punctuation">:</span> joint_idd<span class="token punctuation">,</span> <span class="token string">'bbox'</span><span class="token punctuation">:</span> bbox<span class="token punctuation">}</span>
</code></pre>
<ol start="5">
<li>数据增强<br>
作者用到了几种数据增强的手段</li>
</ol>
<ul>
<li>缩放 scale</li>
<li>旋转 rotate</li>
<li>翻转 flip</li>
<li>添加颜色噪声 add color noise<br><br>
(1)缩放<br>
读入数据后，需要先把大小不一的标注图片统一转换成256 x 256。<br>
对于LSP测试集，作者使用的是图像的中心作为身体的位置，并直接以图像大小来衡量身体大小。数据集里的原图片是大小不一的（原图尺寸存在bbox里），一般采取crop的方法有好几种，比如直接进行crop，然后放大，这样做很明显会有丢失关节点的可能性。也可以先把图片放在中间，然后将图片缩放到目标尺寸范围内原尺寸的可缩放的大小，然后四条边还需要填充的距离，最后resize到应有大小。<br>
这里采用的是先扩展边缘，然后放大图片，再进行crop，这样做能够保证图片中心处理后依然在中心位置，且没有关节因为crop而丢失。注意在处理图片的同时需要对标注也进行处理。<br>
<em><strong>要注意opencv和PIL读入的RGB顺序是不一样的，在使用不同库进行处理的时候要转换通道</strong></em></li>
</ul>
<pre class=" language-python"><code class="prism  language-python"><span class="token keyword">import</span> cv2
big_img <span class="token operator">=</span> cv2<span class="token punctuation">.</span>copyMakeBorder<span class="token punctuation">(</span>img<span class="token punctuation">,</span> add<span class="token punctuation">,</span> add<span class="token punctuation">,</span> add<span class="token punctuation">,</span> add<span class="token punctuation">,</span> borderType <span class="token operator">=</span> cv2<span class="token punctuation">.</span>BORDER_CONSTANT<span class="token punctuation">,</span> value<span class="token operator">=</span>self<span class="token punctuation">.</span>pixel_means<span class="token punctuation">.</span>reshape<span class="token punctuation">(</span><span class="token operator">-</span><span class="token number">1</span><span class="token punctuation">)</span><span class="token punctuation">)</span>
<span class="token comment">#self.show(bimg)  </span>
bbox <span class="token operator">=</span> np<span class="token punctuation">.</span>array<span class="token punctuation">(</span>dic<span class="token punctuation">[</span><span class="token string">'bbox'</span><span class="token punctuation">]</span><span class="token punctuation">)</span><span class="token punctuation">.</span>reshape<span class="token punctuation">(</span><span class="token number">4</span><span class="token punctuation">,</span> <span class="token punctuation">)</span><span class="token punctuation">.</span>astype<span class="token punctuation">(</span>np<span class="token punctuation">.</span>float32<span class="token punctuation">)</span>
bbox<span class="token punctuation">[</span><span class="token punctuation">:</span><span class="token number">2</span><span class="token punctuation">]</span> <span class="token operator">+=</span> add
<span class="token keyword">if</span> <span class="token string">'joints'</span> <span class="token keyword">in</span> dic<span class="token punctuation">:</span>
    process<span class="token punctuation">(</span>joints_anno<span class="token punctuation">)</span>
objcenter <span class="token operator">=</span> np<span class="token punctuation">.</span>array<span class="token punctuation">(</span><span class="token punctuation">[</span>bbox<span class="token punctuation">[</span><span class="token number">0</span><span class="token punctuation">]</span> <span class="token operator">+</span> bbox<span class="token punctuation">[</span><span class="token number">2</span><span class="token punctuation">]</span> <span class="token operator">/</span> <span class="token number">2</span><span class="token punctuation">.</span><span class="token punctuation">,</span> bbox<span class="token punctuation">[</span><span class="token number">1</span><span class="token punctuation">]</span> <span class="token operator">+</span> bbox<span class="token punctuation">[</span><span class="token number">3</span><span class="token punctuation">]</span> <span class="token operator">/</span> <span class="token number">2</span><span class="token punctuation">.</span><span class="token punctuation">]</span><span class="token punctuation">)</span>
minx<span class="token punctuation">,</span> miny<span class="token punctuation">,</span> maxx<span class="token punctuation">,</span> maxy <span class="token operator">=</span> compute<span class="token punctuation">(</span>extend_border<span class="token punctuation">,</span> objcenter<span class="token punctuation">,</span> in_size<span class="token punctuation">,</span> out_size<span class="token punctuation">)</span>
img <span class="token operator">=</span> cv2<span class="token punctuation">.</span>resize<span class="token punctuation">(</span>big_img<span class="token punctuation">[</span>min_y<span class="token punctuation">:</span> max_y<span class="token punctuation">,</span> min_x<span class="token punctuation">:</span> max_x<span class="token punctuation">,</span><span class="token punctuation">:</span><span class="token punctuation">]</span><span class="token punctuation">,</span> <span class="token punctuation">(</span>width<span class="token punctuation">,</span> height<span class="token punctuation">)</span><span class="token punctuation">)</span>
</code></pre>
<p>示例图（左：原图、右：缩放后）：<br>
<img src="https://ai-studio-static-online.cdn.bcebos.com/2824e603256145b38e3c91979d7f9cbf43d5453c40744ff4b08771824043aaff" alt="enter image description here"><img src="https://ai-studio-static-online.cdn.bcebos.com/af00a193923048d3827fee3eca9acd11c95e4d92809441c3bdd1052756bbfee7" alt="enter image description here"><br>
示例图的十四个标注点：<br>
(88.995834, 187.24898)；(107.715065, 160.57408)；(119.648575, 124.30561)<br>
(135.3259, 124.53958)；(145.38748, 155.4263)；(133.68799, 165.95587)<br>
(118.47862, 109.330215)；(108.41703, 104.65042)；(120.81852, 84.05927)<br>
(151.70525, 86.63316)；(162.93677, 101.14057)；(161.29883, 124.773575)<br>
(136.0279, 85.93119)；(138.13379, 66.509995)</p>
<p>(2)旋转<br>
旋转后点的坐标需要通过一个旋转矩阵来确定，在网上的开源代码中，作者使用了以下矩阵的变换矩阵围绕着(x,y)进行任意角度的变换。<br>
<img src="https://ai-studio-static-online.cdn.bcebos.com/e220f5617a9945d6a5abd76634c84314142dbf1fc11244348897fab68e016b5b" alt="enter image description here"><br>
在opencv中可以使用</p>
<pre class=" language-python"><code class="prism  language-python">cv2<span class="token punctuation">.</span>getRotationMatrix2D<span class="token punctuation">(</span><span class="token punctuation">(</span>center_x<span class="token punctuation">,</span> center_y<span class="token punctuation">)</span> <span class="token punctuation">,</span> angle<span class="token punctuation">,</span> <span class="token number">1.0</span><span class="token punctuation">)</span>
newimg <span class="token operator">=</span> cv2<span class="token punctuation">.</span>warpAffine<span class="token punctuation">(</span>img<span class="token punctuation">,</span> rotMat<span class="token punctuation">,</span> <span class="token punctuation">(</span>width<span class="token punctuation">,</span> height<span class="token punctuation">)</span><span class="token punctuation">)</span>
</code></pre>
<p>得到转换矩阵，并通过仿射变换得到旋转后的图像。<br>
而标注点可以直接通过旋转矩阵获得对应点。</p>
<pre class=" language-python"><code class="prism  language-python">rot <span class="token operator">=</span> rotMat<span class="token punctuation">[</span><span class="token punctuation">:</span><span class="token punctuation">,</span> <span class="token punctuation">:</span> <span class="token number">2</span><span class="token punctuation">]</span>
add <span class="token operator">=</span> np<span class="token punctuation">.</span>array<span class="token punctuation">(</span><span class="token punctuation">[</span>rotMat<span class="token punctuation">[</span><span class="token number">0</span><span class="token punctuation">]</span><span class="token punctuation">[</span><span class="token number">2</span><span class="token punctuation">]</span><span class="token punctuation">,</span> rotMat<span class="token punctuation">[</span><span class="token number">1</span><span class="token punctuation">]</span><span class="token punctuation">[</span><span class="token number">2</span><span class="token punctuation">]</span><span class="token punctuation">]</span><span class="token punctuation">)</span>
coor <span class="token operator">=</span> np<span class="token punctuation">.</span>dot<span class="token punctuation">(</span>rot<span class="token punctuation">,</span> coor<span class="token punctuation">)</span> <span class="token operator">+</span> w
</code></pre>
<p><img src="https://ai-studio-static-online.cdn.bcebos.com/4469abf377ef433b967916a955a99d09b1de257d0382401e904bf3e6b622047f" alt="enter image description here"><br>
(3) 翻转<br>
使用opencv中的flip进行翻转，并对标注点进行处理。在opencv中flip函数的参数有1水平翻转、0垂直翻转、-1水平垂直翻转三种</p>
<pre class=" language-python"><code class="prism  language-python"><span class="token keyword">def</span> <span class="token function">flip</span><span class="token punctuation">(</span>self<span class="token punctuation">,</span> img<span class="token punctuation">,</span> cod<span class="token punctuation">,</span> anno_valid<span class="token punctuation">,</span> symmetry<span class="token punctuation">)</span><span class="token punctuation">:</span>
    <span class="token triple-quoted-string string">'''对图片进行翻转'''</span>
    newimg <span class="token operator">=</span> cv2<span class="token punctuation">.</span>flip<span class="token punctuation">(</span>img<span class="token punctuation">,</span> <span class="token number">1</span><span class="token punctuation">)</span>
    train_data<span class="token punctuation">[</span>counter<span class="token punctuation">]</span> <span class="token operator">=</span> newimg<span class="token punctuation">.</span>transpose<span class="token punctuation">(</span><span class="token number">2</span><span class="token punctuation">,</span> <span class="token number">0</span><span class="token punctuation">,</span> <span class="token number">1</span><span class="token punctuation">)</span>
    <span class="token triple-quoted-string string">'''处理标注点，symmetry是flip后所对应的标注，具体需要自己根据实际情况确定'''</span>
    <span class="token keyword">for</span> <span class="token punctuation">(</span>l<span class="token punctuation">,</span> r<span class="token punctuation">)</span> <span class="token keyword">in</span> symmetry<span class="token punctuation">:</span>
        cod<span class="token punctuation">[</span>l<span class="token punctuation">]</span><span class="token punctuation">,</span> cod<span class="token punctuation">[</span>r<span class="token punctuation">]</span> <span class="token operator">=</span> cod<span class="token punctuation">[</span>l<span class="token punctuation">]</span><span class="token punctuation">,</span> cod<span class="token punctuation">[</span>r<span class="token punctuation">]</span>
    <span class="token keyword">for</span> i <span class="token keyword">in</span> <span class="token builtin">range</span><span class="token punctuation">(</span>n<span class="token punctuation">)</span><span class="token punctuation">:</span>
        label<span class="token punctuation">.</span>append<span class="token punctuation">(</span><span class="token punctuation">(</span>cod<span class="token punctuation">[</span>i<span class="token punctuation">]</span><span class="token punctuation">[</span><span class="token number">0</span><span class="token punctuation">]</span><span class="token punctuation">,</span>cod<span class="token punctuation">[</span>i<span class="token punctuation">]</span><span class="token punctuation">[</span><span class="token number">1</span><span class="token punctuation">]</span><span class="token punctuation">)</span><span class="token punctuation">)</span>
    train_label<span class="token punctuation">[</span>cnt<span class="token operator">+</span><span class="token operator">+</span><span class="token punctuation">]</span> <span class="token operator">=</span> np<span class="token punctuation">.</span>array<span class="token punctuation">(</span>label<span class="token punctuation">)</span>
</code></pre>
<p><img src="https://ai-studio-static-online.cdn.bcebos.com/b0369391593d4b599f0687023cb57528f2a728c0e1e84c94978ec3b5d1ba6d24" alt="enter image description here"><br>
(4)添加颜色噪声<br>
我所采用的方法是直接添加10%高斯分布的颜色点作为噪声。人为地损失部分通道的信息也可以达到添加彩色噪声的效果。</p>
<pre class=" language-python"><code class="prism  language-python"><span class="token keyword">def</span> <span class="token function">add_color_noise</span><span class="token punctuation">(</span>self<span class="token punctuation">,</span> image<span class="token punctuation">,</span> percentage<span class="token operator">=</span><span class="token number">0.1</span><span class="token punctuation">)</span><span class="token punctuation">:</span>
    noise_img <span class="token operator">=</span> image 
    <span class="token triple-quoted-string string">'''产生图像大小10%的随机点'''</span>
    num <span class="token operator">=</span> <span class="token builtin">int</span><span class="token punctuation">(</span>percentage<span class="token operator">*</span>image<span class="token punctuation">.</span>shape<span class="token punctuation">[</span><span class="token number">0</span><span class="token punctuation">]</span><span class="token operator">*</span>image<span class="token punctuation">.</span>shape<span class="token punctuation">[</span><span class="token number">1</span><span class="token punctuation">]</span><span class="token punctuation">)</span>
    <span class="token triple-quoted-string string">'''添加噪声'''</span>
    <span class="token keyword">for</span> i <span class="token keyword">in</span> <span class="token builtin">range</span><span class="token punctuation">(</span>num<span class="token punctuation">)</span><span class="token punctuation">:</span> 
        x <span class="token operator">=</span> np<span class="token punctuation">.</span>random<span class="token punctuation">.</span>randint<span class="token punctuation">(</span><span class="token number">0</span><span class="token punctuation">,</span>image<span class="token punctuation">.</span>shape<span class="token punctuation">[</span><span class="token number">0</span><span class="token punctuation">]</span><span class="token punctuation">)</span> 
        y <span class="token operator">=</span> np<span class="token punctuation">.</span>random<span class="token punctuation">.</span>randint<span class="token punctuation">(</span><span class="token number">0</span><span class="token punctuation">,</span>image<span class="token punctuation">.</span>shape<span class="token punctuation">[</span><span class="token number">1</span><span class="token punctuation">]</span><span class="token punctuation">)</span> 
        <span class="token keyword">for</span> j <span class="token keyword">in</span> <span class="token builtin">range</span><span class="token punctuation">(</span><span class="token number">3</span><span class="token punctuation">)</span><span class="token punctuation">:</span>
	        noise_img<span class="token punctuation">[</span>x<span class="token punctuation">,</span> y<span class="token punctuation">,</span> i<span class="token punctuation">]</span> <span class="token operator">=</span> noise_img<span class="token punctuation">[</span>x<span class="token punctuation">,</span> y<span class="token punctuation">,</span> i<span class="token punctuation">]</span> <span class="token operator">+</span> random<span class="token punctuation">.</span>gauss<span class="token punctuation">(</span><span class="token number">2</span><span class="token punctuation">,</span><span class="token number">4</span><span class="token punctuation">)</span>
            noise_img<span class="token punctuation">[</span>x<span class="token punctuation">,</span> y<span class="token punctuation">,</span> i<span class="token punctuation">]</span> <span class="token operator">=</span> <span class="token number">255</span> <span class="token keyword">if</span> noise_img<span class="token punctuation">[</span>x<span class="token punctuation">,</span> y<span class="token punctuation">,</span> ch<span class="token punctuation">]</span> <span class="token operator">&gt;</span> <span class="token number">255</span> <span class="token keyword">else</span> <span class="token number">0</span>
    <span class="token keyword">return</span> noise_img
</code></pre>
<p><img src="https://ai-studio-static-online.cdn.bcebos.com/0b3f37cc24384dceb920942b88e0c799706e6ab80ed14cec8b7df3e791d98342" alt="enter image description here"><br>
(4)除此之外，以下数据增强的方法也很常见：</p>
<ul>
<li>从颜色上考虑，还可以做图像亮度、饱和度、对比度变化、PCA Jittering（按照RGB三个颜色通道计算均值和标准差后在整个训练集上计算协方差矩阵，进行特征分解，得到特征向量和特征值）</li>
<li>从图像空间性质上考虑，还可以使用随机裁剪、平移</li>
<li>从噪声角度，高斯噪声、椒盐噪声、模糊处理；</li>
<li>从类别分布的角度，可以采用label shuffle、Supervised Data Augmentation（海康威视ILSVRC2016的report）<br>
在这个具体例子中，进行数据增强的时候要考虑的是（1）形变会不会影响结果（2）会不会丢掉部分节点</li>
</ul>
<h4 id="制作paddle数据">制作paddle数据</h4>
<p>使用paddle.batch批量读入数据，并制作成paddle的数据格式</p>
<pre class=" language-python"><code class="prism  language-python">reader <span class="token operator">=</span> paddle<span class="token punctuation">.</span>batch<span class="token punctuation">(</span>self<span class="token punctuation">.</span>read_record<span class="token punctuation">(</span>test_list<span class="token punctuation">,</span> joint_dict<span class="token punctuation">,</span> 
				mode <span class="token operator">=</span> <span class="token string">'train'</span><span class="token punctuation">)</span><span class="token punctuation">,</span> batch_size<span class="token operator">=</span><span class="token number">1</span><span class="token punctuation">)</span>
fluid<span class="token punctuation">.</span>recordio_writer<span class="token punctuation">.</span>convert_reader_to_recordio_file<span class="token punctuation">(</span><span class="token string">"./work/test_"</span>  <span class="token operator">+</span> <span class="token builtin">str</span><span class="token punctuation">(</span>i<span class="token punctuation">)</span> <span class="token operator">+</span> <span class="token string">"_test.recordio"</span><span class="token punctuation">,</span> 
                feeder<span class="token operator">=</span>feeder<span class="token punctuation">,</span> reader_creator<span class="token operator">=</span>reader<span class="token punctuation">)</span> 
</code></pre>
<h4 id="其他数据相关内容">其他数据相关内容</h4>
<p>论文的评价标准：<br>
PCK：检测的关键点与其对应的groundtruth之间的归一化距离小于设定阈值的比例。<br>
在本篇论文中，作者将图片中心作为身体的位置，并以图片大小作为衡量身体尺寸的标准。<br>
PCK@0.2 on LSP &amp;&amp; LSP-extended：以驱干直径为归一化标准<br>
PCKh@0.5 on MPII：以头部为归一化标准</p>
<h4 id="关于训练的过拟合抢救">关于训练的过拟合抢救</h4>
<p>对于容易过拟合的数据，数据增强是比较重要的，训练的时候学习率需要不能太大，当一次训练过拟合后，可以从loss曲线波动的地方回溯到较为平稳的点，并以平稳点的学习率为起点，以更低的学习率接上上次学习。</p>

