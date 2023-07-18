featurenames = ['S2_peakAmp', '10_5', '25_10', '50_25', '75_25', \
                '75_50', '90_75', '95_90', 'ext_elec']

display_vars = ['S2_peakAmp', '10_5', '25_10', '50_25', '75_25', \
                '75_50', '90_75', '95_90', 'ext_elec']

palette = {
    "Tritium": 'cyan',
    "gas": '#5a01a4',
    "gate": '#d24f71',
    "cath": '#fdad32',
    "Tritium": 'cyan'
}

sample = df

g = sns.pairplot(sample, vars=display_vars, hue='type', palette=palette)

for feature in featurenames:
    sns.distplot(df_cath[feature].astype('float64'), kde=False, label='Cathode',
                 hist_kws={'weights': df_cath['weights']}, color='#fdad32')
    sns.distplot(df_gate[feature].astype('float64'), kde=False, label='Gate', hist_kws={'weights': df_gate['weights']},
                 color='#d24f71')
    sns.distplot(df_gas[feature].astype('float64'), kde=False, label='Gas', hist_kws={'weights': df_gas['weights']},
                 color='#5a01a4')

# make range diff for last plot
plt.savefig('bigplot.png')
plt.show()
