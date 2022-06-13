from bokeh.client import push_session
from bokeh.io import curdoc
from bokeh.layouts import row
from bokeh.models import ColumnDataSource
from bokeh.models.widgets import DataTable, TableColumn
curdoc().clear()
#replaced `sin(x)` with `x+2`
x=[i for i in range(10)]
def update_y(x):
      y=[]
      for i in x:
            y.append(int(i)+2)
      return y

def update_table(x):
      y=update_y(x)
      data=dict(x=x,y=y)
      source=ColumnDataSource(data)
      columns=[TableColumn(field='x',title='x'),TableColumn(field='y',title='y')]
      data_table=DataTable(source=source,columns=columns,width=500,height=500,editable=True)
      curdoc().clear()
      curdoc().add_root(row(data_table))
      source.on_change('data',update)
def update(attr,old,new):
      update_table(new['x'])

ses=push_session(curdoc())
update_table(x)  
ses.show()
ses.loop_until_closed()
