from bbpipe import PipelineStage
from .types import FitsFile,YamlFile,DummyFile

class BBPowerSpecter(PipelineStage):
    """
    Template for a power spectrum stage
    """
    name="BBPowerSpecter"
    inputs=[('splits_info',YamlFile),('window_function',FitsFile),('nmt_fields',DummyFile)]
    outputs=[('power_spectra_splits',DummyFile),('mode_coupling_matrix',DummyFile)]
    config_options={'bpw_edges':[2,30,50,70,90,110,130,150,170,190,210,230,250,270,290,350,1000],
                    'beam_correct':True}

    def run(self) :
        #This stage currently does nothing whatsoever
        print(self.config)
        for inp,_ in self.inputs :
            fname=self.get_input(inp)
            print("Reading "+fname)
            open(fname)

        for out,_ in self.outputs :
            fname=self.get_output(out)
            print("Writing "+fname)
            open(fname,"w")

if __name__ == '__main__':
    cls = PipelineStage.main()
