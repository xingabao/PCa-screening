# ------------------------------------------------------------------------------
# Load required packages
# ------------------------------------------------------------------------------
suppressMessages(suppressWarnings(library(glue)))
suppressMessages(suppressWarnings(library(dplyr)))
suppressMessages(suppressWarnings(library(ggplot2)))

# ------------------------------------------------------------------------------
# Set working directory
# ------------------------------------------------------------------------------
no <- basename(dirname(rstudioapi::getActiveDocumentContext()$path))
wkdir <- dirname(rstudioapi::getActiveDocumentContext()$path)
setwd(wkdir)

# ------------------------------------------------------------------------------
# Load and preprocess data
# ------------------------------------------------------------------------------
# Read dataset
ert_df <- readxl::read_xlsx('../data/train-test-data.xlsx')

# Filter and transform data
ert_df <- ert_df %>%
  filter(TPSA > 10) %>%
  mutate(
    diagnose = case_when(
      diagnose == 0 ~ 'HC',
      diagnose == 1 ~ 'BPH',
      diagnose == 2 ~ 'PCa'
    )
  ) %>% 
  filter(diagnose %in% c('BPH', 'PCa'))

# Log-transform APOE to normalize its distribution
ert_df$APOE <- log(ert_df$APOE)

# ------------------------------------------------------------------------------
# Spearman Correlation Analysis
# ------------------------------------------------------------------------------
# 1. Overall correlation
alb_cor_all <- cor.test(ert_df$APOE, ert_df$`AR+TREM2+`, method = 'spearman', exact = FALSE)

# 2. Subgroup correlation (BPH)
alb_cor_bph <- cor.test(ert_df$APOE[ert_df$diagnose == 'BPH'], 
                        ert_df$`AR+TREM2+`[ert_df$diagnose == 'BPH'], 
                        method = 'spearman', exact = FALSE)

# 3. Subgroup correlation (PCa)
alb_cor_pca <- cor.test(ert_df$APOE[ert_df$diagnose == 'PCa'], 
                        ert_df$`AR+TREM2+`[ert_df$diagnose == 'PCa'], 
                        method = 'spearman', exact = FALSE)

# Console Output
cat('\n==================================================\n')
cat('       Spearman Correlation Analysis Results      \n')
cat('==================================================\n')
cat(sprintf('Overall:  rho = %.3f, P = %.2e\n', alb_cor_all$estimate, alb_cor_all$p.value))
cat(sprintf('BPH Group: rho = %.3f, P = %.2e\n', alb_cor_bph$estimate, alb_cor_bph$p.value))
cat(sprintf('PCa Group: rho = %.3f, P = %.2e\n', alb_cor_pca$estimate, alb_cor_pca$p.value))
cat('--------------------------------------------------\n')

# Generate Summary Table
alb_summary <- data.frame(
  Group   = c('Overall', 'BPH', 'PCa'),
  n       = c(nrow(ert_df), sum(ert_df$diagnose == 'BPH'), sum(ert_df$diagnose == 'PCa')),
  rho     = round(c(alb_cor_all$estimate, alb_cor_bph$estimate, alb_cor_pca$estimate), 3),
  P_value = signif(c(alb_cor_all$p.value, alb_cor_bph$p.value, alb_cor_pca$p.value), 3)
)
cat('\nSummary Table:\n')
print(alb_summary, row.names = FALSE)

# ------------------------------------------------------------------------------
# Color Palette & Styling Configurations
# ------------------------------------------------------------------------------
# Light fill colors for 95% Confidence Intervals (CI)
rt_fill_ci <- '#F8E6F0'   # Light pink/purple for overall CI
al_fill_ci <- '#EBF5E8'   # Light green/grey for subgroup CI

# Grid lines and text colors
be_grey <- '#404040'
be_grid <- '#D9D9D9'

# Y-axis label with mathematical superscript
alb_ylab <- expression(bold(AR^"+" ~ TREM2^"+" ~ "Monocytes (%)"))

# Dynamic position calculation for the correlation text annotations
x_min <- min(ert_df$APOE, na.rm = TRUE)
x_max <- max(ert_df$APOE, na.rm = TRUE)
y_max <- max(ert_df$`AR+TREM2+`, na.rm = TRUE)

# ------------------------------------------------------------------------------
# Visualization 1: Overall Correlation Plot (Single Trendline)
# ------------------------------------------------------------------------------
gg_overall <- ggplot(ert_df, aes(x = APOE, y = `AR+TREM2+`)) +
  # Linear regression line with 95% CI
  geom_smooth(method = 'lm', se = TRUE, color = be_grey, fill = rt_fill_ci, alpha = 0.45, linewidth = 0.8) +
  # Scatter points
  geom_point(color = '#000000', size = 3, alpha = 0.8) +
  # Dynamic annotation for correlation coefficient
  annotate(
    'text',
    x = x_min + (x_max - x_min) * 0.05,
    y = y_max * 0.95,
    label = sprintf('rho = %.3f\nP %s', 
                    alb_cor_all$estimate, 
                    ifelse(alb_cor_all$p.value < 0.001, '< 0.001', paste('=', signif(alb_cor_all$p.value, 3)))),
    hjust = 0, size = 4.5, family = 'serif',
    color = be_grey, fontface = 'italic'
  ) +
  labs(x = 'Serum APOE (\u03BCg/mL)', y = alb_ylab) +
  # Transform X-axis labels back to raw values from log scale
  scale_x_continuous(labels = function(x) signif(exp(x), 3)) +
  theme_classic(base_size = 11, base_family = 'serif') +
  theme(
    axis.title.x       = element_text(size = 12, color = '#000000', family = 'serif', face = 'bold', margin = margin(t = 10)),
    axis.title.y       = element_text(size = 12, color = '#000000', family = 'serif', face = 'bold', margin = margin(r = 10)),
    axis.text          = element_text(size = 11, color = '#000000', family = 'serif'),
    panel.grid.major   = element_line(color = be_grid, linewidth = 0.25),
    panel.grid.minor   = element_blank(),
    plot.margin        = margin(12, 14, 12, 14)
  )

# ------------------------------------------------------------------------------
# Export Figures
# ------------------------------------------------------------------------------
width <- 5; height <- 4.5

# Save Overall Plot
ggsave(gg_overall, filename = glue('{wkdir}/../Outputs/Fig_correlation/Correlation_Analysis_Overall.png'), width = width, height = height, dpi = 300, bg = '#FFFFFF')
ggsave(gg_overall, filename = glue('{wkdir}/../Outputs/Fig_correlation/Correlation_Analysis_Overall.pdf'), width = width, height = height, dpi = 300, bg = '#FFFFFF')

# ------------------------------------------------------------------------------
# Export Session Info for Reproducibility
# ------------------------------------------------------------------------------
sink(glue('{wkdir}/../Outputs/Fig_correlation/sessionInfo.txt'))
sessionInfo()
sink()
