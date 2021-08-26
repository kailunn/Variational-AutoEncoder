# Variational-AutoEncoder

## 2020Fall - Deep Learning homework

* 2.1 :

`1. Implement VAE and show the learning curve and some reconstructed samples like the given examples. (10%)`
![GITHUB]( https://github.com/kailunn/Variational-AutoEncoder/blob/main/1.png "1")

`2. Sample the prior p(z) and use the latent codes z to synthesize some examples when your model is well-trained. (5%)`
![GITHUB]( https://github.com/kailunn/Variational-AutoEncoder/blob/main/2.png?raw=true "2")
![GITHUB]( https://github.com/kailunn/Variational-AutoEncoder/blob/main/3.png "3")

`3.Show the synthesized images based on the interpolation of two latent codes z between two real samples. (5%)`
![GITHUB]( https://github.com/kailunn/Variational-AutoEncoder/blob/main/4.png "4")
![GITHUB]( https://github.com/kailunn/Variational-AutoEncoder/blob/main/5.png "5")

`4.	Multiply the Kullback-Leiblier (KL) term with a scale λ and tune λ (e.g. λ = 0 and λ = 100) then show the results based on steps 1, 2, 3 and some analyses. (10%)`
=> Total loss = 50% rec + 50% KL divergence loss
當reconstruction loss(λ = 0)比重較大時會較靠近原始圖片
反之則較靠近sample出來的分布所以較不清楚
λ = 100 :
![GITHUB]( https://github.com/kailunn/Variational-AutoEncoder/blob/main/6.png "6")
λ = 0 :
![GITHUB]( https://github.com/kailunn/Variational-AutoEncoder/blob/main/7.png "7")
