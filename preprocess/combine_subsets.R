source("/root/data/asj/2022/sc-transformer-master/preprocess/utils.R")


parser <- ArgumentParser() #创建一个解析对象
parser$add_argument("--task", type = "character", default = "tea_single") #向该对象中添加你要关注的命令行参数和选项
o <- parser$parse_args() #进行解析
# o <- parser$parse_known_args()[[1]]  # for python interactive
task <- o$task

config <- parseTOML("configs/data.toml")[[task]] #parseTOML:解析TOML配置文件  
combs <- config$combs
comb_ratios <- config$comb_ratios
mods_ <- unique(unlist(combs)) #返回一个与x相同类的对象(通常是类似向量、类似数据帧或类似数组的对象)，但删除了重复的元素/行。
mods <- vector()
for (mod in c("atac", "rna", "adt")) {
    if (mod %in% mods_) {
        mods <- c(mods, mod)
    }
}

input_dirs <- pj(config$raw_data_dirs, "seurat")
task_dir <- pj("data", "processed", task)
output_fig_dir <- pj(task_dir, "fig")
mkdir(output_fig_dir, remove_old = F)
output_feat_dir <- pj(task_dir, "feat")
mkdir(output_feat_dir, remove_old = F)



merge_counts <- function(mod) {

    # Load different subsets
    prt("Processing ", mod, " data ...\n")
    sc_list <- list()
    feat_list <- list()
    subset_id <- "0"
    for (dataset_id in seq_along(input_dirs)) {
        comb <- combs[[dataset_id]]
        comb_ratio <- comb_ratios[[dataset_id]]
        fp <- pj(input_dirs[dataset_id], paste0(mod, ".h5seurat"))
        if (file.exists(fp)) {
            prt("Loading ", fp, " ...\n")
            sc <- LoadH5Seurat(fp) #从 h5Seurat 文件加载保存的"Seurat"对象,[features, samples]
            cell_num <- dim(sc)[2]
            end_ids <- round(cumsum(comb_ratio) / sum(comb_ratio) * cell_num) #round:四舍五入
            start_ids <- c(1, end_ids + 1)
            for (split_id in seq_along(comb)) {
                if (mod %in% comb[[split_id]]) {
                    cell_names <- colnames(sc)[start_ids[split_id]:end_ids[split_id]]
                    sc_list[[subset_id]] <- subset(sc, cells = cell_names)
                    sc_list[[subset_id]]$subset_id <- subset_id
                    feat_list[[subset_id]] <- rownames(sc_list[[subset_id]])
                }
                subset_id <- toString(strtoi(subset_id) + 1) #Convert Strings to Integers
            }
        }
        else {
            subset_id <- toString(strtoi(subset_id) + length(comb))
        }
    }
    feat_union <- Reduce(union, feat_list) #reduce：累积迭代函数

    # # debugging for adt
    # map(feat_list, length)
    # fl <- feat_list[c("0", "4", "8")]
    # map(fl, length)

    # Reduce(setdiff, feat_list[c("0", "4")])
    # Reduce(setdiff, feat_list[c("4", "0")])

    # Reduce(setdiff, fl[c("0", "8")])
    # Reduce(setdiff, fl[c("8", "0")])

    # Reduce(setdiff, fl[c("4", "8")])
    # Reduce(setdiff, fl[c("8", "4")])

    # str_sort(unlist(fl[1]))
    # str_sort(unlist(fl[2]))

    # str_extract(unlist(fl[1]), pattern = "^cd3$")
    # str_extract(unlist(fl[2]), pattern = "cd56.*")
    # str_extract(unlist(fl[3]), pattern = "^cd3$")

    # length(Reduce(intersect, feat_list[c("0", "4")]))
    # length(Reduce(union, feat_list[c("0", "4")]))

    # length(Reduce(intersect, fl[c("0", "8")]))
    # length(Reduce(union, fl[c("0", "8")]))

    # length(Reduce(intersect, fl[c("4", "8")]))
    # length(Reduce(union, fl[c("4", "8")]))


    # Remove low-frequency features
    mask_list <- list()
    mask_sum_list <- list()
    cell_num_total <- 0
    for (subset_id in names(feat_list)) {
        mask_list[[subset_id]] <- as.integer(feat_union %in% feat_list[[subset_id]]) #as.integer用于将字符对象转换为整数对象
        cell_num <- dim(sc_list[[subset_id]])[2]
        mask_sum_list[[subset_id]] <- mask_list[[subset_id]] * cell_num
        cell_num_total <- cell_num_total + cell_num
    }
    mask_sum_total <- Reduce(`+`, mask_sum_list)
    mask_ratio <- mask_sum_total / cell_num_total
    feat_union <- feat_union[mask_sum_total > 5000 | mask_ratio > 0.5]#条件概率


    # Find highly variable features
    var_feat_list <- list()
    for (subset_id in names(sc_list)) {
        sc_list[[subset_id]] <- subset(sc_list[[subset_id]], features = feat_union)
        if (mod == "rna") {
            sc_list[[subset_id]] <- FindVariableFeatures(sc_list[[subset_id]], nfeatures = 2000) #
        } else if (mod == "adt") {
            VariableFeatures(sc_list[[subset_id]]) <- rownames(sc_list[[subset_id]])
        } else {
            stop(paste0(mod, ": Invalid modality"))
        }
        var_feat_list[[subset_id]] <- VariableFeatures(sc_list[[subset_id]])
    }

    if (mod == "rna") {
        var_feat_integ <- SelectIntegrationFeatures(sc_list, nfeatures = 2000)
    } else {
        var_feat_integ <- Reduce(union, var_feat_list)
    }

    # Select features for each subset
    sc_list <- lapply(sc_list, subset, features = var_feat_integ)

    # Merge different subsets
    subset_num <- length(sc_list)
    if (subset_num > 1) {
        sc_merge <- merge(sc_list[[1]], unlist(sc_list[2:subset_num]),
            add.cell.ids = paste0("B", names(sc_list)), merge.data = T)
    } else {
        sc_merge <- RenameCells(sc_list[[1]], add.cell.id = paste0("B", names(sc_list)[1]))
    }
    feat_merged <- rownames(sc_merge)
    rownames(sc_merge[[mod]]@counts) <- feat_merged  # correct feature names for count data
    feat_dims[[mod]] <<- length(feat_merged)
    write.csv(feat_merged, file = pj(output_feat_dir, paste0("feat_names_", mod, ".csv")))

    # Get feature masks for each subset
    mask_list <- list()
    for (subset_id in names(sc_list)) {
        mask_list[[subset_id]] <- as.integer(feat_merged %in% rownames(sc_list[[subset_id]]))
    }

    # Split into subsets and save
    sc_split <- SplitObject(sc_merge, split.by = "subset_id")
    for (subset_id in names(sc_split)) {
        prt("Saving subset ", subset_id, " ...\n")
        output_dir <- pj(task_dir, paste0("subset_", subset_id))
        output_mat_dir <- pj(output_dir, "mat")
        mkdir(output_mat_dir, remove_old = F)

        mat <- t(data.matrix(sc_split[[subset_id]][[mod]]@counts))  # N * D
        # Save count data
        write.csv(mat, file = pj(output_mat_dir, paste0(mod, ".csv")))
        # Save cell IDs
        write.csv(rownames(mat), file = pj(output_dir, "cell_names.csv"))

        output_mask_dir <- pj(output_dir, "mask")
        mkdir(output_mask_dir, remove_old = F)
        mask <- t(data.matrix(mask_list[[subset_id]]))  # 1 * D
        # Save the feature mask
        write.csv(mask, file = pj(output_mask_dir, paste0(mod, ".csv")))
    }
}



merge_frags <- function() {
    mod <- "atac"
    # Load different subsets
    prt("Processing ", mod, " data ...\n")
    sc_list <- list()
    feat_list <- list()

    subset_id <- "0"
    for (dataset_id in seq_along(input_dirs)) {
        comb <- combs[[dataset_id]]
        comb_ratio <- comb_ratios[[dataset_id]]
        fp <- pj(input_dirs[dataset_id], paste0(mod, ".h5seurat"))
        if (file.exists(fp)) {
            prt("Loading ", fp, " ...\n")
            sc <- LoadH5Seurat(fp)
            cell_num <- dim(sc)[2]
            end_ids <- round(cumsum(comb_ratio) / sum(comb_ratio) * cell_num)
            start_ids <- c(1, end_ids + 1)
            for (split_id in seq_along(comb)) {
                if (mod %in% comb[[split_id]]) {
                    cell_names <- colnames(sc)[start_ids[split_id]:end_ids[split_id]]
                    sc_list[[subset_id]] <- subset(sc, cells = cell_names)
                    feat_list[[subset_id]] <- StringToGRanges(rownames(sc_list[[subset_id]]))
                }
                subset_id <- toString(strtoi(subset_id) + 1)
            }
        }
        else {
            subset_id <- toString(strtoi(subset_id) + length(comb))
        }
    }
    feat_merged <- Signac::reduce(do.call("c", unname(feat_list))) #unname:从对象中删除名称或名称,但是好像没变化啊

    # Filter out bad peaks based on length
    feat_widths <- width(feat_merged)
    feat_merged <- feat_merged[feat_widths < 10000 & feat_widths > 20]
    feat_merged

    # Re-compute peak counts based on merged features
    for (subset_id in names(sc_list)) {
        dataset_id <- sum(strtoi(subset_id) >= c(0, cumsum(lengths(combs))))
        frag_path <- pj(config$raw_data_dirs[dataset_id], config$raw_data_frags[dataset_id])
        cell_names <- colnames(sc_list[[subset_id]])
        cell_names_copy <- NULL
        if (grepl("TEA-seq", frag_path)) {
            cell_names_copy <- cell_names
            metadata <- read.csv(file = gsub("fragments.tsv", "metadata.csv", frag_path))
            cell_names <- metadata$barcodes[match(cell_names, metadata$original_barcodes)]
        }
        frags <- CreateFragmentObject(path = frag_path, cells = cell_names)
        counts <- FeatureMatrix(fragments = frags, features = feat_merged, cells = cell_names) ## quantify multiome peaks in the scATAC-seq dataset
        assay <- CreateChromatinAssay(counts = counts, fragments = frags) ## create object
        sc_list[[subset_id]] <- CreateSeuratObject(counts = assay, assay = "atac")
        sc_list[[subset_id]]$subset_id <- subset_id
        if (grepl("TEA-seq", frag_path)) {
            sc_list[[subset_id]] <- RenameCells(sc_list[[subset_id]], new.names = cell_names_copy)
        }
    }

    # Remove low-frequency features for each subset
    var_feat_list <- list()
    for (subset_id in names(sc_list)) {
        # sc_list[[subset_id]] <- FindTopFeatures(sc_list[[subset_id]], min.cutoff = "q75")
        cell_num <- dim(sc_list[[subset_id]])[2]
        feat_ratio <- rowSums(sc_list[[subset_id]]$atac@counts > 0) / cell_num
        hist(feat_ratio, xlim = range(0, 1), breaks = seq(0, 1, l = 300))
        var_feat_list[[subset_id]] <- rownames(sc_list[[subset_id]])[feat_ratio > 0.04]
    }
    var_feat_union <- Reduce(union, var_feat_list)

    # Select features for each subset
    sc_list <- lapply(sc_list, subset, features = var_feat_union)

    # Merge different subsets
    subset_num <- length(sc_list)
    if (subset_num > 1) {
        sc_merge <- merge(sc_list[[1]], unlist(sc_list[2:subset_num]),
            add.cell.ids = paste0("B", names(sc_list)), merge.data = T)
    } else {
        sc_merge <- RenameCells(sc_list[[1]], add.cell.id = paste0("B", names(sc_list)[1]))
    }
    feat_merged <- rownames(sc_merge)
    rownames(sc_merge[[mod]]@counts) <- feat_merged  # correct feature names for count data
    # sort features
    gr_sorted <- sort(StringToGRanges(feat_merged))
    feat_dims[[mod]] <<- width(gr_sorted@seqnames)
    feat_sorted <- GRangesToString(gr_sorted)
    write.csv(feat_sorted, file = pj(output_feat_dir, paste0("feat_names_", mod, ".csv")))

    # Split into subsets and save
    sc_split <- SplitObject(sc_merge, split.by = "subset_id")
    for (subset_id in names(sc_split)) {
        prt("Saving subset ", subset_id, " ...\n")
        output_dir <- pj(task_dir, paste0("subset_", subset_id))
        output_mat_dir <- pj(output_dir, "mat")
        mkdir(output_mat_dir, remove_old = F)

        mat <- t(data.matrix(sc_split[[subset_id]][[mod]]@counts)[feat_sorted, ])  # N * D
        # Save count data
        write.csv(mat, file = pj(output_mat_dir, paste0(mod, ".csv")))
        # Save cell IDs
        write.csv(rownames(mat), file = pj(output_dir, "cell_names.csv"))
    }
}


feat_dims <- list()
for (mod in mods) {
    if (mod == "atac") {
        merge_frags()
    } else {
        merge_counts(mod)
    }
}
# Save feature dimensionalities
prt("feat_dims: ", feat_dims, "\n")
write.csv(feat_dims, file = pj(output_feat_dir, "feat_dims.csv"))