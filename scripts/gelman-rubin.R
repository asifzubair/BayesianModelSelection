df = list()
for (i in c(1,3,4,5,6,7,8,10)) { 
    df[[i]] = read.table(file.path(paste0("run_", str_pad(i, 2, pad = "0")), "posteriors_pt_1M_10.csv"), header = T)
    df[[i]] = df[[i]][seq(300000, 1000000, 10), "LnLike"]
}

df = list()
for (i in seq(10)) { 
    df[[i]] = read.table(file.path(paste0("run_", str_pad(i, 2, pad = "0")), "posteriors_pt_1M_10.csv"), header = T)
    df[[i]] = df[[i]][seq(300000, 1000000, 10), "LnLike"]
}

tmp = lapply(df, mcmc)
tmp = lapply(c(df[[1]], df[[3]], df[[4]], df[[5]], df[[6]], df[[7]], df[[8]], df[[10]]), mcmc)

tmp = mcmc.list(mcmc(df[[1]]), mcmc(df[[3]]), mcmc(df[[4]]), mcmc(df[[5]]), 
                mcmc(df[[6]]), mcmc(df[[7]]), mcmc(df[[8]]), mcmc(df[[10]]))
combined.chains = as.mcmc.list(tmp)
gelman.plot(combined.chains)
gelman.plot(combined.chains, main = "Model B7r")#, xlab = "", ylab = "")

# for (i in seq(10)) { 
#     print(i)
#     df[[i]] = mcmc(df[[i]][seq(300000, 1000000, 10), ])
# }
# 
# 
# mcmc.list(df[[1]], df[[2]], df[[3]], df[[4]], df[[5]], df[[6]], df[[7]], df[[8]], df[[9]], df[[10]])
# mcmc.list(df[[1]], df[[3]], df[[4]], df[[5]], df[[6]], df[[7]], df[[8]], df[[10]])