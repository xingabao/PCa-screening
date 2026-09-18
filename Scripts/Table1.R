# Load R packages
suppressMessages(suppressWarnings(library(gt)))
suppressMessages(suppressWarnings(library(dplyr)))
suppressMessages(suppressWarnings(library(gtsummary)))

# Set Env
wkdir <- dirname(rstudioapi::getActiveDocumentContext()$path)
setwd(wkdir)

# Load data
dat.raw <- readxl::read_excel('../data/dat.final.xlsx')
A = c('Urea', 'AST', 'ALT', 'CREA', 'PDW', 'NEUT%', 'LY%', 'MONO%', 'MPV', 'BASO%', 'MONO#', 'NEUT#', 'EO#', 'LY#', 'MCHC', 'RDW-CV', 'PLT', 'HGB', 'MCH', 'MCV', 'HCT', 'BASO#', 'AFP', 'APOE', 'CEA', 'APOE2', 'AR-TREM2-', 'AR-TREM2+', 'AR+TREM2+', 'AR+TREM2-', 'FPSA/TPSA', 'TPSA', 'FPSA', 'age')
C = c('Group', 'diagnose', 'age', 'AR+TREM2+', 'AR+TREM2-', 'AR-TREM2+', 'AR-TREM2-', 'FPSA', 'TPSA', 'FPSA/TPSA', 'CEA' , 'AFP', 'APOE', 'MCHC', 'RDW-CV', 'HCT', 'BASO#', 'MCH', 'MCV', 'HGB', 'PLT', 'NEUT%', 'LY%', 'MPV', 'EO#', 'MONO%', 'BASO%', 'LY#', 'MONO#', 'NEUT#', 'PDW', 'CREA', 'ALT', 'AST', 'Urea', 'γ-GT')
setdiff(A, C)
setdiff(C, A)

# Arrange data
dat. <- dat.raw %>% 
  select(c('Group', 'diagnose', 'age', 'AR+TREM2+', 'AR+TREM2-', 'AR-TREM2+', 'AR-TREM2-', 'FPSA', 'TPSA', 'FPSA/TPSA', 'CEA' , 'AFP', 'APOE', 'MCHC', 'RDW-CV', 'HCT', 'BASO#', 'MCH', 'MCV', 'HGB', 'PLT', 'NEUT%', 'LY%', 'MPV', 'EO#', 'MONO%', 'BASO%', 'LY#', 'MONO#', 'NEUT#', 'PDW', 'CREA', 'ALT', 'AST', 'Urea')) %>%
  mutate(Group = case_when(
    Group == 'GZ' ~ 'Derivation and testing',
    Group == 'AH' ~ 'External validation',
    TRUE ~ NA
  ))

table(dat.$Group)

dat.$`AR-TREM2-` <- dat.$`AR-TREM2-` * 100
dat.$`AR-TREM2+` <- dat.$`AR-TREM2+` * 100
dat.$`AR+TREM2-` <- dat.$`AR+TREM2-` * 100
dat.$`AR+TREM2+` <- dat.$`AR+TREM2+` * 100

# Convert the cohort identification column to a factor and set the order to ensure correct table column ordering
dat.$Group <- factor(
  dat.$Group,
  levels = c("Derivation and testing", "External validation")
)

# Create summary table
summary_table <- dat. %>%
  # `by = Group` is key, as it creates a column for each cohort
  tbl_summary(
    by = Group,
    # Define statistical data display format
    statistic = list(
      all_continuous() ~ "{mean} ± {sd} ({min}–{max})", # Matches the original image format
      all_categorical(dichotomous = FALSE) ~ "{n} ({p}%)", # Multi-categorical variables
      all_dichotomous() ~ "{n} ({p}%)" # Binary variables
    ),
    # Define decimal places
    digits = list(
      all_continuous() ~ 2, # Age as integer (or formatted to 2 decimals)
      all_categorical() ~ 1 # Percentage kept to one decimal place
    ),
    # For binary variables, we only care about the "Yes" case, so hide "No"
    value = all_dichotomous() ~ "Yes",
    # Do not show missing value statistics
    missing = "no"
  ) %>%
  # Modify column headers
  modify_header(
    label = "**Characteristic**", # Set the header of the first column to "Characteristic"
    stat_1 = "**Derivation and testing cohort (n=518)**",
    stat_2 = "**External validation cohort (n=151)**"
  ) %>%
  # Convert the gtsummary object to a gt object for finer style adjustments
  as_gt() %>%
  # Use gt package features for beautification
  tab_header(
    title = md("**Table 1 | Baseline demographic and clinical characteristics**")
  ) %>%
  # Adjust styles to match the original image
  tab_options(
    table.border.top.color = "black",
    table.border.top.width = px(2.5),
    column_labels.border.bottom.color = "black",
    column_labels.border.bottom.width = px(2.5),
    table_body.border.bottom.color = "black",
    table_body.border.bottom.width = px(2.5),
    table.font.size = px(14),
    heading.title.font.size = px(18),
    column_labels.font.weight = "bold"
  ) %>%
  # Set background color and style for row group headers
  tab_style(
    style = list(
      cell_fill(color = "#F0F0E0"), # A beige color close to the original image
      cell_text(weight = "bold")
    ),
    locations = cells_row_groups()
  ) %>%
  # Remove gt default row group borders
  opt_row_striping(row_striping = FALSE)


# Print or display the final table
summary_table

# Export the table to Word
# Convert the table to a flextable and save it as a Word document
if (FALSE) { gtsave(summary_table, filename = '../Outputs/table1.docx') }