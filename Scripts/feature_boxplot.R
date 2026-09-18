# Load R packages
suppressMessages(suppressWarnings(library(glue)))
suppressMessages(suppressWarnings(library(dplyr)))
suppressMessages(suppressWarnings(library(ggplot2)))
suppressMessages(suppressWarnings(library(ggbreak)))
suppressMessages(suppressWarnings(library(ggsignif)))
suppressMessages(suppressWarnings(library(patchwork)))
suppressMessages(suppressWarnings(library(ggVennDiagram)))

# 
generate_bounded_ticks <- function(start_num, end_num, n = 4) {
  # 1. Calculate the maximum allowed step size
  # If we take a step larger than this, the (n)th number will exceed end_num
  max_step <- (end_num - start_num) / (n - 1)
  
  # 2. Determine the magnitude (power of 10)
  exponent <- floor(log10(max_step))
  magnitude <- 10^exponent
  
  # 3. Define "nice" factors to check (standard chart intervals)
  # We check them in descending order to find the largest one that fits.
  # You can add 1.5, 3, 4, etc., if you want more flexible intervals.
  factors <- c(5, 2.5, 2, 1)
  
  # Default step (in case loop fails, though it shouldn't for these factors)
  step_factor <- 1
  
  for (f in factors) {
    if (f * magnitude <= max_step) {
      step_factor <- f
      break
    }
  }
  
  # 4. Calculate the final nice step
  nice_step <- step_factor * magnitude
  
  # Handle edge case where step might calculate to 0 (if range is 0)
  if (nice_step == 0) return(rep(start_num, n))
  
  # 5. Generate the sequence
  ticks <- seq(from = start_num, by = nice_step, length.out = n)
  
  return(ticks)
}

# Set Env
wkdir <- dirname(rstudioapi::getActiveDocumentContext()$path)
setwd(wkdir)

# Load data
dat.tt <- readxl::read_xlsx('../data/train-test-data.xlsx')
dat.vn <- readxl::read_xlsx('../data/validation-data.xlsx')

#
allcols <- c('Urea', 'AST', 'ALT', 'CREA', 'PDW', 'NEUT%', 'LY%', 'MONO%', 'MPV', 'BASO%', 'MONO#', 'NEUT#', 'EO#', 'LY#', 'MCHC', 'RDW-CV', 'PLT', 'HGB', 'MCH', 'MCV', 'HCT', 'BASO#', 'AFP', 'APOE', 'CEA', 'APOE2', 'AR-TREM2-', 'AR-TREM2+', 'AR+TREM2+', 'AR+TREM2-', 'FPSA/TPSA', 'TPSA', 'FPSA', 'age')

my.color <- c('#19A955', '#404040', '#E50611')

#
gg.plot <- function(col, dat, breaks = NULL, heights = c(1, 1)) {
  
  dat. <- dat %>% select(all_of(c('diagnose', col)))
  
  # Calculate median and quartiles
  quantiles <- dat. %>%
    group_by(diagnose) %>%
    summarize(
      q25 = quantile(get(col), 0.25),
      median = quantile(get(col), 0.5),
      q75 = quantile(get(col), 0.75))
  
  # CORRECTED FUNCTIONd
  map_signif_level <- function(p) {
    # ggsignif 可能会一次传入多个 p 值，所以我们需要向量化处理
    # 使用 sapply 确保对每个 p 值单独处理
    sapply(p, function(single_p) {
      if (single_p > 0.05) {
        # 大于 0.05，保留三位小数
        p_val <- sprintf("%.3f", single_p)
        label <- paste0("italic(P) == '", p_val, "'") # 使用 == 而不是 ~"="~ 更符合 plotmath 语法
      } else if (single_p < 0.001) {
        # 小于 0.001，使用科学计数法
        p_str <- formatC(single_p, format = "e", digits = 2) # digits=2 对应 %.2e，总共3位有效数字
        p_parts <- unlist(strsplit(p_str, "e"))
        base <- p_parts[1]
        exponent <- as.integer(p_parts[2])
        # 构造 plotmath 字符串: P = base * 10^exponent
        label <- paste0("italic(P) == ", base, " %*% 10^", exponent)
      } else {
        # 介于 0.001 和 0.05 之间，保留三位小数
        p_val <- sprintf("%.3f", single_p)
        label <- paste0("italic(P) == '", p_val, "'")
      }
      return(label)
    })
  }
  
  # Calculate positions
  y.pos.1 <- max(dat.[, col]) * 1.00
  y.pos.2 <- max(dat.[, col]) * 1.20
  y.pos.3 <- max(dat.[, col]) * 1.40
  y.max <- max(dat.[, col]) * 1.80
  y.min <- 0
  y.label <- max(dat.[, col]) * 1.68
  
  # Define the variable name label
  var_label <- ifelse(col %in% names(convert), convert[[col]], col)
  
  # Base Plot Constructiond
  gg.dbase <- ggplot(dat., aes(x = diagnose, y = get(col))) +
    geom_boxplot(aes(fill = diagnose), alpha = 0.1, linewidth = 0.5) +
    geom_point(aes(color = diagnose), alpha = 0.1, position = position_jitter(width = 0.1), size = 1.5, stroke = 1.5) +
    geom_errorbar(data = quantiles, aes(x = diagnose, y = median, ymin = q25, ymax = q75, color = diagnose), width = 0.15, linewidth = 1.0) +
    geom_crossbar(data = quantiles, aes(x = diagnose, y = median, ymin = median, ymax = median, color = diagnose), width = 0.6, linewidth = 0.50) +
    scale_fill_manual(values = my.color) +
    scale_color_manual(values = my.color) +
    theme_test(base_size = 9) +
    labs(y = col, x = NULL) +
    theme(
      strip.background = element_rect(fill = 'white', color = 'white'),
      axis.text.x = element_blank(),
      axis.text.y = element_text(size = 9, color = '#000000'),
      axis.title.x = element_blank(),
      axis.title.y = element_blank(),
      axis.ticks.x = element_blank(),
      legend.position = 'none'
    )
  
  if (is.null(breaks)) {
    
    b <- generate_bounded_ticks(y.min, y.max, 7)[1:(length(generate_bounded_ticks(y.min, y.max, 7)) - 1)]
    
    # STANDARD PLOT (No Break)
    gg <- gg.dbase +
      geom_signif(
        comparisons = list(c('Healthy', 'BPH'), c('Healthy', 'PCa'), c('BPH', 'PCa')), 
        map_signif_level = map_signif_level,
        y_position = c(y.pos.1, y.pos.2, y.pos.3), 
        textsize = 2.5, tip_length = 0.02,
        parse = TRUE
      ) +
      annotate('text', x = 0.5, y = y.label, label = var_label, size = 2.5, color = '#000000', hjust = 0, vjust = 0, fontface = 'bold') +
      scale_y_continuous(limits = c(y.min, y.max), expand = c(0, 0), breaks = b) +
      theme(panel.border = element_rect(color = '#000000', linewidth = 1))
    
    return(gg)
    
  } else {
    
    # BROKEN AXIS PLOT (Patchwork)d
    low_break <- breaks[1]
    high_break <- breaks[2]
    
    b <- generate_bounded_ticks(high_break, y.max)[1:(length(generate_bounded_ticks(high_break, y.max)) - 1)]
    
    # 1. Top Plot
    gg.top <- gg.dbase +
      geom_signif(
        comparisons = list(c('Healthy', 'BPH'), c('Healthy', 'PCa'), c('BPH', 'PCa')), 
        vjust = -0.10,
        map_signif_level = map_signif_level,
        y_position = c(y.pos.1, y.pos.2, y.pos.3), 
        textsize = 2.5, tip_length = 0.02,
        parse = TRUE
      ) +
      annotate('text', x = 0.5, y = y.label, label = var_label, size = 2.5, color = '#000000', hjust = 0, vjust = 0, fontface = 'bold') +
      scale_y_continuous(sec.axis = sec_axis(~ ., labels = NULL, name = NULL), expand = c(0, 0), breaks = b) +
      coord_cartesian(ylim = c(high_break, y.max), clip = 'on') + 
      scale_x_discrete(position = 'top') + 
      theme(
        panel.border = element_blank(), 
        axis.line.y.left = element_line(color = '#000000', linewidth = 0.5), 
        axis.line.x.top = element_line(color = '#000000', linewidth = 0.5),  
        axis.ticks.y.right = element_blank(), 
        axis.line.y.right = element_line(color = '#000000', linewidth = 0.5),
        axis.line.x.bottom = element_blank(), 
        axis.ticks.x = element_blank(),
        plot.margin = margin(t = 0, b = 5, r = 0, l = 0) 
      )
    
    # 2. Bottom Plot
    if (heights[1] > heights[2]) {
      gg.bottom <- gg.dbase +
        scale_y_continuous(sec.axis = sec_axis(~ ., labels = NULL, name = NULL), expand = c(0, 0), breaks = c(0, low_break))
    } else {
      gg.bottom <- gg.dbase +
        scale_y_continuous(sec.axis = sec_axis(~ ., labels = NULL, name = NULL), expand = c(0, 0))
    }
    gg.bottom <- gg.bottom +
      coord_cartesian(ylim = c(y.min, low_break), clip = 'on') +
      theme(
        panel.border = element_blank(),
        axis.line.y.left = element_line(color = '#000000', linewidth = 0.5), 
        axis.line.x.bottom = element_line(color = '#000000', linewidth = 0.5), 
        axis.ticks.y.right = element_blank(),
        axis.line.y.right = element_line(color = '#000000', linewidth = 0.5), 
        axis.line.x.top = element_blank(), 
        plot.title = element_blank(),
        plot.margin = margin(t = 5, b = 0, r = 0, l = 0)
      )
    
    # 3. Combine
    gg <- gg.top / gg.bottom + plot_layout(heights = heights)
    
    return(gg)
  }
}

#
convert <- list()
convert['FPSA'] <- 'fPSA (µg/L)'
convert['TPSA'] <- 'tPSA (µg/L)'
convert['FPSA/TPSA'] <- 'fPSA/tPSA (ratio)'
convert['LY%'] <- 'LY (%)'
convert['HCT'] <- 'HCT (%)'
convert['RDW-CV'] <- 'RDW-CV (%)'
convert['Urea'] <- 'Urea (mmol/L)'
convert['HGB'] <- 'HGB (g/L)'
convert['LY#'] <- 'LY# (*10^9/L)'
convert['TPSA*AR+TREM2+'] <- 'TPSA*AR+TREM2+ (µg/L)'
convert['NEUT#'] <- 'NEUT# (*10^9/L)'
convert['MONO#'] <- 'MONO# (*10^9/L)'
convert['PLT'] <- 'PLT (*10^9/L)'
convert['MCHC'] <- 'MCHC (g/L)'
convert['APOE*AR+TREM2+'] <- 'APOE*AR+TREM2+ (mg/L)'
convert['MCH'] <- 'MCH (pg)'
convert['AFP*AR+TREM2+'] <- 'AFP*AR+TREM2+ (µg/L)'
convert['AST'] <- 'AST (U/L)'
convert['ALT'] <- 'ALT (U/L)'
convert['CREA'] <- 'CREA (µmol/L)'
convert['PDW'] <- 'PDW (fL)'
convert['NEUT%'] <- 'NEUT (%)'
convert['MONO%'] <- 'MONO (%)'
convert['MPV'] <- 'MPV (fL)'
convert['BASO#'] <- 'BASO# (*10^9/L)'
convert['EO#'] <- 'EO# (*10^9/L)'
convert['MCV'] <- 'MCV (fL)'
convert['BASO%'] <- 'BASO (%)'
convert['AFP'] <- 'AFP (µg/L)'
convert['APOE'] <- 'APOE (mg/L)'
convert['CEA'] <- 'CEA (µg/L)'
convert['AR-TREM2-'] <- 'AR-TREM2-'
convert['AR-TREM2+'] <- 'AR-TREM2+'
convert['AR+TREM2-'] <- 'AR+TREM2-'
convert['AR+TREM2+'] <- 'AR+TREM2+'
convert['age'] <- 'Age (year)'

#
breaks.list <- list()
breaks.list[['MAIN-1-MODEL-FPSA']] <- c(6, 600)
breaks.list[['MAIN-1-MODEL-TPSA']] <- c(40, 1000)
breaks.list[['MAIN-1-MODEL-HCT']] <- c(10, 25)
breaks.list[['MAIN-1-MODEL-RDW-CV']] <- c(8, 8)
breaks.list[['MAIN-1-MODEL-Urea']] <- c(15, 35)
breaks.list[['MAIN-1-MODEL-HGB']] <- c(200, 450)
breaks.list[['MAIN-1-MODEL-TPSA*AR+TREM2+']] <- c(20, 1000)
breaks.list[['MAIN-1-MODEL-MONO#']] <- c(1, 4)
breaks.list[['MAIN-1-MODEL-MCHC']] <- c(250, 250)
breaks.list[['MAIN-1-MODEL-APOE*AR+TREM2+']] <- c(100, 600)
breaks.list[['MAIN-1-MODEL-MCH']] <- c(15, 15)
breaks.list[['MAIN-1-MODEL-AFP*AR+TREM2+']] <- c(3, 8)
breaks.list[['SUP-1-MODEL-FPSA']] <- c(5, 40)
breaks.list[['SUP-1-MODEL-TPSA']] <- c(30, 300)
breaks.list[['SUP-1-MODEL-FPSA/TPSA']] <- c(0.7, 1.2)
breaks.list[['SUP-1-MODEL-HCT']] <- c(10, 25)
breaks.list[['SUP-1-MODEL-RDW-CV']] <- c(10, 10)
breaks.list[['SUP-1-MODEL-HGB']] <- c(80, 80)
breaks.list[['SUP-1-MODEL-TPSA*AR+TREM2+']] <- c(20, 400)
breaks.list[['SUP-1-MODEL-MONO#']] <- c(1, 4)
breaks.list[['SUP-1-MODEL-MCHC']] <- c(250, 250)
breaks.list[['SUP-1-MODEL-APOE*AR+TREM2+']] <- c(100, 600)
breaks.list[['SUP-1-MODEL-MCH']] <- c(20, 20)
breaks.list[['SUP-1-MODEL-AFP*AR+TREM2+']] <- c(5, 10)
breaks.list[['SUP-2-MODEL-AST']] <- c(50, 250)
breaks.list[['SUP-2-MODEL-ALT']] <- c(80, 210)
breaks.list[['SUP-2-MODEL-CREA']] <- c(200, 650)
breaks.list[['SUP-2-MODEL-PDW']] <- c(7, 7)
breaks.list[['SUP-2-MODEL-NEUT%']] <- c(25, 25)
breaks.list[['SUP-2-MODEL-MONO%']] <- c(20, 30)
breaks.list[['SUP-2-MODEL-MPV']] <- c(6, 6)
breaks.list[['SUP-2-MODEL-EO#']] <- c(1, 2)
breaks.list[['SUP-2-MODEL-MCV']] <- c(50, 50)
breaks.list[['SUP-2-MODEL-CEA']] <- c(15, 75)
breaks.list[['SUP-3-MODEL-APOE']] <- c(200, 2000)
breaks.list[['SUP-3-MODEL-MPV']] <- c(7, 7)
breaks.list[['SUP-3-MODEL-MCV']] <- c(50, 50)
breaks.list[['SUP-3-MODEL-BASO%']] <- c(1, 3)
breaks.list[['SUP-3-MODEL-BASO#']] <- c(0.1, 0.2)
#
small.heights <- c(
  'MAIN-1-MODEL-HCT', 'MAIN-1-MODEL-RDW-CV', 'MAIN-1-MODEL-MCHC', 'MAIN-1-MODEL-MCH',
  'SUP-1-MODEL-HCT', 'SUP-1-MODEL-RDW-CV', 'SUP-1-MODEL-HGB', 'SUP-1-MODEL-MCHC', 'SUP-1-MODEL-MCH',
  'SUP-2-MODEL-PDW', 'SUP-2-MODEL-NEUT%', 'SUP-2-MODEL-MPV', 'SUP-2-MODEL-MCV',
  'SUP-3-MODEL-MPV', 'SUP-3-MODEL-MCV'
)

# 
for (this in c('MAIN-1-MODEL', 'SUP-1-MODEL', 'SUP-2-MODEL', 'SUP-3-MODEL')) {
# for (this in c('SUP-3-MODEL')) {
  if (this %in% c('MAIN-1-MODEL', 'SUP-2-MODEL')) {
    dat <- dat.tt
  } else {
    dat <- dat.vn
  }
  
  dat <- dat %>% mutate(
    diagnose = case_when(
      diagnose == 0 ~ 'Healthy',
      diagnose == 1 ~ 'BPH',
      diagnose == 2 ~ 'PCa',
      TRUE ~ ''
    )
  )
  
  dat$diagnose <- factor(dat$diagnose, levels = c('Healthy', 'BPH', 'PCa'))
  
  col.base <- c('FPSA', 'TPSA', 'FPSA/TPSA')
  col.model1 <- c('TPSA', 'LY%', 'HCT', 'RDW-CV', 'FPSA/TPSA', 'FPSA', 'Urea', 'HGB', 'LY#', 'TPSA*AR+TREM2+', 'NEUT#', 'MONO#', 'PLT')
  col.model2 <- c('TPSA', 'FPSA/TPSA', 'NEUT#', 'MCHC', 'APOE*AR+TREM2+', 'MCH', 'AFP*AR+TREM2+', 'age')
  
  cols <- unique(c(col.base, col.model1, col.model2))
  cols.ot <- setdiff(allcols, c(cols, 'APOE2'))
  
  gg.list <- list()
  for (col in cols) {
    this.id <- glue('{this}-{col}')
    if (this.id %in% names(breaks.list)) {
      if (this.id %in% small.heights) {
        gg.list[[col]] = gg.plot(col, dat, breaks = breaks.list[[this.id]], heights = c(7, 1))
      } else {
        gg.list[[col]] = gg.plot(col, dat, breaks = breaks.list[[this.id]])
      }
    } else {
      gg.list[[col]] = gg.plot(col, dat)
    }
  }
  
  hh.list <- list()
  for (col in cols.ot) {
    this.id <- glue('{this}-{col}')
    if (this.id %in% names(breaks.list)) {
      if (this.id %in% small.heights) {
        hh.list[[col]] = gg.plot(col, dat, breaks = breaks.list[[this.id]], heights = c(7, 1))
      } else {
        hh.list[[col]] = gg.plot(col, dat, breaks = breaks.list[[this.id]])
      }
    } else {
      hh.list[[col]] = gg.plot(col, dat)
    }
  }

  # 制作图例
  groups <- c('Healthy Individual', 'Benign Prostatic\nHyperplasia (BPH)', 'Prostate Cancer\n(PCa)')
  
  legend.df <- data.frame(
    x = 1,
    y = 3:1,
    Group = factor(groups, levels = groups),
    Color = my.color
  )
  
  gg.legend <- ggplot(legend.df, aes(x = x, y = y)) +
    geom_point(aes(color = Group), shape = 15, size = 4) + 
    geom_text(aes(label = Group), hjust = 0, nudge_x = 0.40,  fontface = 'bold', size = 3) +
    scale_color_manual(values = setNames(my.color, groups)) +
    scale_x_continuous(limits = c(-0.5, 6), expand = c(0, 0)) + 
    scale_y_continuous(limits = c(.0, 5.0), expand = c(0, 0)) +
    theme_void() +
    theme(
      legend.position = 'none',
      plot.background = element_rect(fill = '#FFFFFF', color = NA),
      plot.margin = margin(t = 0, r = 0, b = 0, l = -40)
    )
  
  # 添加韦恩图
  if (this == 'MAIN-1-MODEL') {
    
    remr <- function(strings) {
      strings <- gsub("FPSA", "fPSA", strings)
      strings <- gsub("TPSA", "tPSA", strings)
      strings <- gsub("FPSA/TPSA", "fPSA/tPSA", strings)
      return(strings)
    }
    
    sl <- function(strings) {
      sorted_strings <- strings[order(nchar(strings))]
      return(sorted_strings)
    }
    
    x <- list(
      A = col.base,
      B = col.model1,
      C = col.model2
    )
    
    venn.dat <- ggVennDiagram::Venn(x)
    venn.dat <- ggVennDiagram::process_data(venn.dat)
    venn.dat <- ggVennDiagram::venn_region(venn.dat)
    ABC. <- venn.dat %>% filter(name == 'A/B/C') %>% pull(item) %>% unlist() %>% remr() %>% sl()
    A. <- venn.dat %>% filter(name == 'A') %>% pull(item) %>% unlist() %>% remr() %>% sl()
    B. <- venn.dat %>% filter(name == 'B') %>% pull(item) %>% unlist() %>% remr() %>% sl()
    C. <- venn.dat %>% filter(name == 'C') %>% pull(item) %>% unlist() %>% remr() %>% sl()
    
    gg.ven <- ggVennDiagram(
      x,  
      category.names = c('                                       Canonical Prostate-specific Antigens\n\n\n\n', 'Biomarkers for the 1st Model\n\n', '\n\nBiomarkers for the 2nd Model'),
      show_intersect = FALSE, 
      set_color = '#000000',
      set_size = 3,
      label_size = 3,
      label = 'count',
      label_alpha = 0,
      label_font = 'sans',
      label_geom = 'label',
      label_color = '#000000',
      label_percent_digit = 2,
      label_txtWidth = 60,
      edge_lty = 'dashed',
      relative_width = 0.3,
      shape_id = '301f',
    ) +
      coord_fixed(clip = 'off') +
      scale_fill_gradient(low = '#FFFFFF', high = '#b9292b', name = 'Count') +
      guides(fill = 'none')
    
    gg.ven <- gg.ven + ggplot2::annotate(geom = 'text', x = 9.5, y = -6.5, label = paste0(ABC., collapse = '\n'), color = '#E50914', lineheight = 1.00, size = 2.0)
    gg.ven <- gg.ven + ggplot2::annotate(geom = 'text', x = -5.5, y = -5.5, label = paste0(C., collapse = '\n'), color = '#000000', lineheight = 1.00, size = 2.0)
    gg.ven <- gg.ven + ggplot2::annotate(geom = 'text', x = 11.5, y = -1.0, label = paste0(B., collapse = '\n'), color = '#000000', lineheight = 1.00, size = 2.0)
    gg.ven <- gg.ven + ggplot2::annotate(geom = 'text', x = -2.0, y = 2.0, label = paste0(A., collapse = '\n'), color = '#E50914', lineheight = 1.00, size = 2.0)
    
    gg.ven <- gg.ven + geom_segment(aes(x = -3.5, y = -5.5, xend = 0.0, yend = -6.5), arrow = arrow(length = unit(0.12, "cm")), color = '#000000')
    gg.ven <- gg.ven + geom_segment(aes(x = 10.0, y = 0.0, xend = 6.5, yend = 0.5), arrow = arrow(length = unit(0.12, "cm")), color = '#000000')
    gg.ven <- gg.ven + geom_segment(aes(x = 7.5, y = -6.0, xend = 2.0, yend = -2.5), arrow = arrow(length = unit(0.12, "cm")), color = '#000000')
    gg.ven <- gg.ven + geom_segment(aes(x = -4.0, y = 7.0, xend = -2.0, yend = 2.8), arrow = arrow(length = unit(0.12, "cm")), color = '#000000', linetype = 'dashed')
    
    gg.list[['venn']] <- gg.ven + theme(plot.margin = margin(l = 20, t = 3))
  }
  
  gg.list[['']] <- gg.legend
  hh.list[['']] <- gg.legend
  
  # Save to file
  if (this %in% c('MAIN-1-MODEL', 'SUP-1-MODEL')) {
    width = 11; height = 2.4 * 4
    
    for (i in 1:(length(gg.list) - 1)) {
      tag.label <- paste0('(', LETTERS[i], ')') 
      gg.list[[i]] <- wrap_elements(gg.list[[i]]) + labs(tag = tag.label)
    }
    
    gg <- wrap_plots(gg.list, ncol = 5, heights = c(1, 1, 1, 1)) + 
      plot_annotation() &
      theme(
        plot.tag = element_text(
          size = 9,
          face = 'bold',
          family = 'sans', margin = margin(t = 0, b = -15, r = -15)
        ),
        plot.margin = margin(t = 0, r = 0, b = 2, l = 0) 
      )
    
    ggsave(gg, filename = glue('{wkdir}/../Outputs/Fig_boxplot/{this}.pdf'), width = width, height = height, bg = '#FFFFFF')
    ggsave(gg, filename = glue('{wkdir}/../Outputs/Fig_boxplot/{this}.png'), width = width, height = height, dpi = 300, device = 'png', bg = '#FFFFFF')
  } else {
    width = 11; height = 2.4 * 4
    
    for (i in 1:(length(hh.list) - 1)) {
      tag.label <- paste0('(', LETTERS[i], ')') 
      hh.list[[i]] <- wrap_elements(hh.list[[i]]) + labs(tag = tag.label)
    }
    
    hh <- wrap_plots(hh.list, ncol = 5, heights = c(1, 1, 1, 1)) + 
      plot_annotation() &
      theme(
        plot.tag = element_text(
          size = 9,
          face = 'bold',
          family = 'sans', margin = margin(t = 0, b = -15, r = -15)
        ),
        plot.margin = margin(t = 0, r = 0, b = 2, l = 0) 
      )
    
    ggsave(hh, filename = glue('{wkdir}/../Outputs/Fig_boxplot/{this}.pdf'), width = width, height = height, bg = '#FFFFFF')
    ggsave(hh, filename = glue('{wkdir}/../Outputs/Fig_boxplot/{this}.png'), width = width, height = height, dpi = 300, device = 'png', bg = '#FFFFFF')
  }
}

# sessionInfo
sink(glue('{wkdir}/sessionInfo.txt'))
sessionInfo()
sink()